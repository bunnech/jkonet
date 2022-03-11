#!/usr/bin/python3
# author: Charlotte Bunne

# imports
import jax
import jax.numpy as jnp
import numpy as np
import optax

# internal imports
from jkonet.utils.helper import count_parameters
from jkonet.utils.optim import global_norm, penalize_weights_icnn
from jkonet.models import fixpoint_loop
from jkonet.models.loss import sinkhorn_loss


def get_step_fn(optimize_psi_fn, psi, optimizer_psi, teacher_forcing=True,
                cumulative=False, parallel=False, epsilon=0.1,
                loss='sinkhorn', train=True):
    """Create a one-step training and evaluation function of Energy."""

    def loss_fn_energy(params_energy, rng_psi, batch, t):
        # initialize psi model and optimizer
        params_psi = psi.init(
            rng_psi, jnp.ones(batch[t].shape[1]))['params']
        opt_state_psi = optimizer_psi.init(params_psi)

        # solve jko step
        _, predicted, loss_psi = optimize_psi_fn(
            params_energy, params_psi, opt_state_psi, batch[t])

        # compute sinkhorn distance between prediction and data
        if loss == 'sinkhorn':
            loss_energy = sinkhorn_loss(predicted, batch[t + 1], epsilon,
                                        div=True)
        elif loss == 'wasserstein':
            loss_energy = sinkhorn_loss(predicted, batch[t + 1], epsilon,
                                        div=False)
        else:
            raise NotImplementedError

        return loss_energy, (loss_psi, predicted)

    def loss_fn_energy_cum(params_energy, rng_psi, batch):
        # iterate through time steps
        def _through_time(batch, t):
            # initialize psi model and optimizer
            params_psi = psi.init(
                rng_psi, jnp.ones(batch[t].shape[1]))['params']
            opt_state_psi = optimizer_psi.init(params_psi)

            # solve jko step
            _, predicted, loss_psi = optimize_psi_fn(
                params_energy, params_psi, opt_state_psi, batch[t])

            # compute sinkhorn distance between prediction and data
            if loss == 'sinkhorn':
                loss_energy = sinkhorn_loss(predicted, batch[t + 1], epsilon,
                                            div=True)
            elif loss == 'wasserstein':
                loss_energy = sinkhorn_loss(predicted, batch[t + 1], epsilon,
                                            div=False)
            else:
                raise NotImplementedError

            # if no teacher-forcing, replace next overvation with predicted
            batch = jax.lax.cond(
                teacher_forcing, lambda x: x,
                lambda x: jax.ops.index_update(x, t + 1, predicted), batch)

            return batch, (loss_energy, loss_psi, predicted)

        _, (loss_energy, loss_psi, predicted) = jax.lax.scan(
            _through_time, batch, jnp.arange(len(batch) - 1))

        return jnp.sum(loss_energy), (loss_energy, loss_psi, predicted)

    @jax.jit
    def step_fn_cum(inputs, batch):
        """Running one step of training or evaluation with cumulative loss."""
        rng_psi, state_energy = inputs

        # adjust dimensions
        if parallel:
            rng_psi = jnp.squeeze(rng_psi)

        # define gradient function
        grad_fn_energy = jax.value_and_grad(
            loss_fn_energy_cum, argnums=0, has_aux=True)

        if train:
            # compute gradient
            (loss_energy, (_, loss_psi, _)
             ), grad_energy = grad_fn_energy(
                 state_energy.params, rng_psi, batch)
            if parallel:
                grad_energy = jax.lax.pmean(grad_energy, axis_name="batch")

            # apply gradient to energy optimizer
            state_energy = state_energy.apply_gradients(grads=grad_energy)

            # compute gradient norm
            grad_norm = global_norm(grad_energy)

            return (rng_psi, state_energy), (loss_energy, loss_psi, grad_norm)
        else:
            (loss_energy, (_, _, predicted)), _ = grad_fn_energy(
              state_energy.params, rng_psi, batch)

            return loss_energy, predicted

    @jax.jit
    def step_fn(inputs, batch):
        """Running one step of training or evaluation."""
        rng_psi, state_energy = inputs

        # adjust dimensions
        if parallel:
            rng_psi = jnp.squeeze(rng_psi)

        # define gradient function
        grad_fn_energy = jax.value_and_grad(
            loss_fn_energy, argnums=0, has_aux=True)

        if train:
            # iterate through time steps
            def _through_time(inputs, t):
                state_energy, batch = inputs

                # compute gradient
                (loss_energy, (loss_psi, predicted)
                 ), grad_energy = grad_fn_energy(state_energy.params,
                                                 rng_psi, batch, t)
                if parallel:
                    grad_energy = jax.lax.pmean(grad_energy, axis_name="batch")

                # apply gradient to energy optimizer
                state_energy = state_energy.apply_gradients(grads=grad_energy)

                # compute gradient norm
                grad_norm = global_norm(grad_energy)

                # if no teacher-forcing, replace next overvation with predicted
                batch = jax.lax.cond(
                    teacher_forcing, lambda x: x,
                    lambda x: jax.ops.index_update(x, t + 1, predicted), batch)

                return ((state_energy, batch),
                        (loss_energy, loss_psi, grad_norm))

            # iterate through time steps
            (state_energy, _), (
                loss_energy, loss_psi, grad_norm) = jax.lax.scan(
                    _through_time, (state_energy, batch),
                    jnp.arange(len(batch) - 1))

            loss_energy = jnp.sum(loss_energy)

            return (rng_psi, state_energy), (loss_energy, loss_psi, grad_norm)
        else:
            # iterate through time steps
            def _through_time(inputs, t):
                state_energy, batch = inputs

                (loss_energy, (loss_psi, predicted)), _ = grad_fn_energy(
                    state_energy.params, rng_psi, batch, t)

                # if no teacher-forcing, replace next overvation with predicted
                batch = jax.lax.cond(
                    teacher_forcing, lambda x: x,
                    lambda x: jax.ops.index_update(x, t + 1, predicted), batch)

                return ((state_energy, batch),
                        (loss_energy, loss_psi, predicted))

            # iterate through time steps
            (_, _), (loss_energy, loss_psi, predicted) = jax.lax.scan(
              _through_time, (state_energy, batch),
              jnp.arange(len(batch) - 1))

            loss_energy = jnp.sum(loss_energy)

            # do not update state
            return loss_energy, predicted

    if cumulative:
        return step_fn_cum
    else:
        return step_fn


def get_optimize_psi_fn(optimizer_psi, psi, energy, tau=1.0, n_iter=100,
                        min_iter=50, max_iter=200, inner_iter=10,
                        threshold=1e-5, beta=1.0, pos_weights=True, cvx_reg=.0,
                        fploop=False):
    """Create a training function of Psi."""

    def loss_fn_psi(params_psi, params_energy, data):
        grad_psi_data = jax.vmap(lambda x: jax.grad(
            psi.apply, argnums=1)({'params': params_psi}, x))(data)
        predicted = cvx_reg * data + grad_psi_data

        # jko objective
        loss_e = energy.apply(
            {'params': params_energy}, predicted)
        loss_p = jnp.sum(jnp.square(predicted - data))
        loss = loss_e + 1 / tau * loss_p

        # add penalty to negative icnn weights in relaxed setting
        if not pos_weights:
            penalty = penalize_weights_icnn(params_psi)
            loss += beta * penalty

        return loss, grad_psi_data

    @jax.jit
    def step_fn_fpl(params_energy, params_psi, opt_state_psi, data):
        def cond_fn(iteration, constants, state):
            """Condition function for optimization of convex potential Psi.
            """
            _, _ = constants
            _, _, _, _, grad = state

            norm = sum(jax.tree_util.tree_leaves(
                jax.tree_map(jnp.linalg.norm, grad)))
            norm /= count_parameters(grad)

            return jnp.logical_or(iteration == 0,
                                  jnp.logical_and(jnp.isfinite(norm),
                                                  norm > threshold))

        def body_fn(iteration, constants, state, compute_error):
            """Body loop for gradient update of convex potential Psi.
            """
            params_energy, data = constants
            params_psi, opt_state_psi, loss_psi, predicted, _ = state

            (loss_jko, predicted), grad_psi = jax.value_and_grad(
                loss_fn_psi, argnums=0, has_aux=True)(
                    params_psi, params_energy, data)

            # apply optimizer update
            updates, opt_state_psi = optimizer_psi.update(
                grad_psi, opt_state_psi)
            params_psi = optax.apply_updates(params_psi, updates)

            loss_psi = jax.ops.index_update(
                loss_psi, jax.ops.index[iteration // inner_iter], loss_jko)
            return params_psi, opt_state_psi, loss_psi, predicted, grad_psi

        # create empty vectors for losses and predictions
        loss_psi = jnp.full(
            (np.ceil(max_iter / inner_iter).astype(int)), 0., dtype=float)
        predicted = jnp.zeros_like(data, dtype=float)

        # define states and constants
        state = params_psi, opt_state_psi, loss_psi, predicted, params_psi
        constants = params_energy, data

        # iteratively _ psi
        params_psi, _, loss_psi, predicted, _ = fixpoint_loop.fixpoint_iter(
            cond_fn, body_fn, min_iter, max_iter, inner_iter, constants, state)

        return params_psi, predicted, loss_psi

    @jax.jit
    def step_fn(params_energy, params_psi, opt_state_psi, data):
        # iteratively optimize psi
        def apply_psi_update(state_psi, i):
            params_psi, opt_state_psi = state_psi

            # compute gradient of jko step
            (loss_psi, predicted), grad_psi = jax.value_and_grad(
                loss_fn_psi, argnums=0, has_aux=True)(
                    params_psi, params_energy, data)

            # apply optimizer update
            updates, opt_state_psi = optimizer_psi.update(
                grad_psi, opt_state_psi)
            params_psi = optax.apply_updates(params_psi, updates)

            return (params_psi, opt_state_psi), (loss_psi, predicted)

        (params_psi, _), (loss_psi, predicted) = jax.lax.scan(
            apply_psi_update, (params_psi, opt_state_psi), jnp.arange(n_iter))
        return params_psi, predicted[-1], loss_psi

    if fploop:
        return step_fn_fpl
    else:
        return step_fn
