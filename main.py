#!/usr/bin/python3
# author: Charlotte Bunne

# imports
import os
import argparse
import flax
import jax
import jax.numpy as jnp
import yaml
import ml_collections
import time
import wandb
from tqdm import tqdm
from flax.training import checkpoints

# internal imports
from jkonet.data import potential_dataloader
from jkonet.data import trajectory_dataloader
from jkonet.models.model_jko import get_step_fn, get_optimize_psi_fn
from jkonet.networks.energies import SimpleEnergy
from jkonet.networks.icnns import ICNN
from jkonet.utils.optim import create_train_state, get_optimizer
from jkonet.utils import plotting
from jkonet.utils.helper import flat_dict, nest_dict, merge
from jkonet.models.loss import sinkhorn_loss


def run_jko(config, task_dir, logging):
    # set random key
    rng = jax.random.PRNGKey(int(time.time()))
    rng, rng_e, rng_p = jax.random.split(rng, 3)

    # get datasets
    if config.dataset.type == 'trajectory':
        dataloader = trajectory_dataloader.TrajectoryDynamicsDataset(
            config.dataset.name, config.train.batch_size,
            config.dataset.time_steps, config.dataset.missing_values,
            config.dataset.noise)
        dim_data = 2
    elif config.dataset.type == 'potential':
        dataloader = potential_dataloader.PotentialDynamicsDataset(
            config.dataset.name, config.train.batch_size,
            config.dataset.setting)
        dim_data = 2
    else:
        raise NotImplementedError('Dataloader not implemented.')

    # initialize models and optimizers
    if config.energy.model.name == 'simple':
        model_energy = SimpleEnergy(config.energy.model.layers)
    else:
        raise NotImplementedError(
            f'Optimizer {config.energy.model.name} not supported yet!')
    model_psi = ICNN(dim_hidden=config.psi.model.layers,
                     init_fn=config.psi.model.init_fn,
                     pos_weights=config.psi.model.pos_weights)

    optimizer_energy = get_optimizer(config.energy)
    optimizer_psi = get_optimizer(config.psi)

    state_energy = create_train_state(
        rng_e, model_energy, optimizer_energy, dim_data)

    # initialize psi optimization
    optimize_psi_fn = get_optimize_psi_fn(
        optimizer_psi, model_psi, model_energy,
        config.settings.tau, config.psi.optim.n_iter,
        config.psi.optim.min_iter, config.psi.optim.max_iter,
        config.psi.optim.inner_iter, config.psi.optim.thr,
        config.psi.optim.beta, config.psi.model.pos_weights,
        config.settings.cvx_reg, config.settings.fploop)

    # define train and evaln step functions of energy
    train_step_fn = get_step_fn(
        optimize_psi_fn, model_psi, optimizer_psi,
        config.settings.teacher_forcing, config.settings.cumulative,
        config.settings.parallel, config.settings.epsilon,
        loss=config.train.loss, train=True)
    evaln_step_fn = get_step_fn(
        optimize_psi_fn, model_psi, optimizer_psi,
        False, config.settings.cumulative, config.settings.parallel,
        config.settings.epsilon, loss=config.train.loss, train=False)

    # pmap (and jit-compile) multiple training steps for faster running
    if config.settings.parallel:
        p_train_step_fn = jax.pmap(train_step_fn, axis_name='batch')
        p_evaln_step_fn = jax.pmap(evaln_step_fn, axis_name='batch')
        state_energy = flax.jax_utils.replicate(state_energy)

    # init loss tracker
    loss_hist = jnp.inf

    # must be divisible by the number of steps jitted together
    assert config.train.logs_freq % config.train.n_jit_steps == 0 and \
           config.train.eval_freq % config.train.n_jit_steps == 0 and \
           config.train.plot_freq % config.train.n_jit_steps == 0, \
           "Missing logs or checkpoints!"

    # execute training
    for step in tqdm(range(0, config.train.n_iters + 1,
                           config.train.n_jit_steps - 1)):

        if config.settings.parallel:
            # get train batch
            batch = jnp.array([next(dataloader.train()) for _ in range(
                jax.local_device_count())])

            # get random seed for psi initialization
            rng_p, *rng_i = jax.random.split(
                rng_p, num=jax.local_device_count() + 1)
            rng_i = jnp.asarray(rng_i)

            # execute training step
            (_, state_energy), (loss_energy, loss_psi, grad_norm
                                ) = p_train_step_fn(
                                    (rng_i, state_energy), batch)
            loss_energy = flax.jax_utils.unreplicate(loss_energy).mean()

        else:
            # get train batch
            batch = next(dataloader.train())

            # get random seed for psi initialization
            rng_p, rng_i = jax.random.split(rng_p, num=2)

            # execute training step
            (_, state_energy), (loss_energy, loss_psi, grad_norm
                                ) = train_step_fn((rng_i, state_energy), batch)

        if logging:
            # log to wandb
            if step % config.train.logs_freq == 0:
                wandb.log({'train_loss_energy': float(loss_energy),
                           'grad_norm_energy': float(jnp.sum(grad_norm)),
                           'step': step})

                for i in loss_psi[0]:
                    wandb.log({'train_loss_psi': float(i)})

        # report the loss on an evaluation dataset periodically
        if (step != 0 and step % config.train.eval_freq == 0):

            if config.settings.parallel:
                # get train batch
                eval_batch = jnp.array(
                    [next(dataloader.evaln()) for _ in range(
                        jax.local_device_count())])

                # get random seed for psi initialization
                rng_p, *rng_i = jax.random.split(
                    rng_p, num=jax.local_device_count() + 1)
                rng_i = jnp.asarray(rng_i)

                # execute evaluation step
                eval_loss_energy, predicted = p_evaln_step_fn(
                    (rng_i, state_energy), eval_batch)
                eval_loss_energy = flax.jax_utils.unreplicate(
                    eval_loss_energy).mean()
            else:
                # get eval batch
                eval_batch = next(dataloader.evaln())

                # get random seed for psi initialization
                rng_p, rng_i = jax.random.split(rng_p, num=2)

                # execute evaluation step
                eval_loss_energy, predicted = evaln_step_fn(
                    (rng_i, state_energy), eval_batch)

            if logging:
                # log to wandb
                wandb.log({'evaln_loss_energy': float(eval_loss_energy)})

            # save checkpoint
            if (loss_hist >= jnp.sum(eval_loss_energy)
               and jnp.sum(eval_loss_energy) >= 0):
                loss_hist = jnp.sum(eval_loss_energy)
                if logging:
                    # log to wandb
                    wandb.log({'best_evaln_loss_energy': float(eval_loss_energy)})

                loss_hist_eval = 0

                for t in range(len(eval_batch) - 1):
                    loss_hist_eval += sinkhorn_loss(
                      predicted[t], eval_batch[t + 1],
                      config.settings.epsilon, div=True)
                if logging:
                    # log to wandb
                    wandb.log(
                        {'best_evaln_loss_energy': float(loss_hist_eval)})

                if config.settings.parallel:
                    save_state = flax.jax_utils.unreplicate(state_energy)
                else:
                    save_state = state_energy

                checkpoints.save_checkpoint(
                    task_dir, save_state, step, keep=100)

        # generate and save samples
        if (step != 0 and step % config.train.plot_freq == 0
           or step == config.train.n_iters):

            # plot predictions and log to wandb
            plotting.plot_predictions(predicted, eval_batch)
            if isinstance(model_energy, SimpleEnergy):
                if config.settings.parallel:
                    plotting.plot_energy_potential(
                      flax.jax_utils.unreplicate(state_energy), eval_batch[0])
                else:
                    plotting.plot_energy_potential(state_energy, eval_batch)


def main(args):
    # get configuration
    config = yaml.load(
        open(os.path.join(args.config_folder, 'config_base_jko.yaml')),
        yaml.UnsafeLoader)
    config_task = yaml.load(open(
        os.path.join(args.config_folder, f'task/config_{args.task}.yaml')),
        yaml.UnsafeLoader)
    config = merge(config, config_task)

    # init wandb
    if args.wandb:
        wandb.init(project='exps_jkonet', dir=args.out_dir,
                   group=args.exp_group, config=flat_dict(config))
        wandb.run.name = wandb.run.id
        config = nest_dict(wandb.config)
    config = ml_collections.ConfigDict(config)

    if wandb:
        task_dir = wandb.run.dir
    else:
        # create outdir if it does not exist
        task_dir = os.path.join(args.out_dir, args.task)
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)

    # run method
    print('Started run.', flush=True)
    run_jko(config, task_dir=task_dir, logging=args.wandb)
    print('Finished run.', flush=True)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='',
                        help='Path to output directory.')
    parser.add_argument('--config_folder', type=str, default='configs',
                        help='Folder containing the config files.')
    parser.add_argument('--task', type=str, default='styblinski',
                        help='Name of task.')
    parser.add_argument('--wandb', action='store_true',
                        help='Option to run with activated wandb.')
    parser.add_argument('--exp_group', type=str, default='',
                        help='Name of run.')
    parser.add_argument('--debug', action='store_true',
                        help='Option to run in debug mode.')
    args = parser.parse_args()

    # set debug mode
    if args.debug:
        print('Running in DEBUG mode.')
        jax.config.update('jax_disable_jit', True)

    main(args)
