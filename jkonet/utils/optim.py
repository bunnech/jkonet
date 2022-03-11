#!/usr/bin/python3
# author: Charlotte Bunne

# imports
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax.core import freeze
import optax


def get_optimizer(config):
    """Returns a flax optimizer object based on `config`."""

    if config.optim.optimizer == 'Adam':
        optimizer = optax.adam(learning_rate=config.optim.lr,
                               b1=config.optim.beta1, b2=config.optim.beta2,
                               eps=config.optim.eps)
    elif config.optim.optimizer == 'SGD':
        optimizer = optax.sgd(learning_rate=config.optim.lr,
                              momentum=None, nesterov=False)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    if config.optim.grad_clip:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.optim.grad_clip),
            optimizer)
    return optimizer


def create_train_state(rng, model, optimizer, input):
    """Creates initial `TrainState`."""
    params = model.init(rng, jnp.ones(input))['params']
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer)


def global_norm(updates):
    """Compute the global norm across a nested structure of tensors."""
    return jnp.sqrt(
        sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(updates)]))


def clip_weights_icnn(params):
    params = params.unfreeze()
    for k in params.keys():
        if (k.startswith('Wz')):
            params[k]['kernel'] = jnp.clip(params[k]['kernel'], a_min=0)

    return freeze(params)


def penalize_weights_icnn(params):
    penalty = 0
    for k in params.keys():
        if (k.startswith('Wz')):
            penalty += jnp.linalg.norm(jax.nn.relu(-params[k]['kernel']))
    return penalty
