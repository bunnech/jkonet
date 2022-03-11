#!/usr/bin/python3
# author: Charlotte Bunne

# imports
import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import wandb


def plot_psi_field(state_psi, data, name='', save_loc=None):
    # set max and min values
    x_min = np.amin(data, axis=0)[:, 0].min() - 2.0
    x_max = np.amax(data, axis=0)[:, 0].max() + 2.0

    y_min = np.amin(data, axis=0)[:, 1].min() - 2.0
    y_max = np.amax(data, axis=0)[:, 1].max() + 2.0

    # create grid
    x, y = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))
    grid = np.vstack((x.ravel(), y.ravel())).T
    if data.shape[2] > 2:
        grid = np.concatenate((grid, np.zeros((grid.shape[0],
                                               data.shape[2]-2))), axis=1)
    # get energy predictions
    grad_psi_data = jax.vmap(lambda x: jax.grad(state_psi.apply_fn, argnums=1)(
        state_psi.params, x))(grid)

    for i in range(grad_psi_data.shape[0]):
        # plot energy predictions
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
        ax.grid(False)

        ax.quiver(grid[:, 0], grid[:, 1], grad_psi_data[i][:, 0],
                  grad_psi_data[i][:, 1], color='k')
        fig.tight_layout()

        if save_loc is None:
            # plot in summary writer
            wandb.log({name + "psi_field":
                       [wandb.Image(fig, caption="Psi Field")]})
            plt.close('all')
        else:
            # save plot
            fig.savefig(save_loc + '_{}.pdf'.format(i))


def plot_energy_potential(state_energy, data, name='', save_loc=None):
    # set max and min values
    x_min = np.amin(data, axis=0)[:, 0].min() - 2.0
    x_max = np.amax(data, axis=0)[:, 0].max() + 2.0

    y_min = np.amin(data, axis=0)[:, 1].min() - 2.0
    y_max = np.amax(data, axis=0)[:, 1].max() + 2.0

    # create grid
    x, y = np.meshgrid(np.linspace(x_min, x_max, 20),
                       np.linspace(y_min, y_max, 20))
    grid = np.vstack((x.ravel(), y.ravel())).T
    if data.shape[2] > 2:
        grid = np.concatenate((grid, np.zeros((grid.shape[0],
                                               data.shape[2]-2))), axis=1)

    # get energy predictions
    pred = state_energy.apply_fn({'params': state_energy.params}, grid, False)
    z = pred.reshape(x.shape)

    # plot energy predictions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
    ax.grid(False)

    ax.contour(x, y, z, levels=15, linewidths=.5, linestyles='dotted',
               colors='k')
    ctr = ax.contourf(x, y, z, levels=15, cmap='Blues')

    fig.colorbar(ctr, ax=ax)
    fig.tight_layout()

    if save_loc is None:
        # plot in summary writer
        wandb.log({name + 'energy':
                   [wandb.Image(fig, caption="Energy Potential")]})
        plt.close('all')
    else:
        # save plot
        fig.savefig(save_loc + '.pdf')


def plot_predictions(predicted, data, name='', save_loc=None):
    # set max and min values
    x_min = np.amin(data, axis=0)[:, 0].min() - 2.0
    x_max = np.amax(data, axis=0)[:, 0].max() + 2.0

    y_min = np.amin(data, axis=0)[:, 1].min() - 2.0
    y_max = np.amax(data, axis=0)[:, 1].max() + 2.0

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    c_data = clr.LinearSegmentedColormap.from_list(
        'Greys', ['#E5E7E9', '#D7DBDD', '#CACFD2', '#BDC3C7', '#A6ACAF'],
        N=data.shape[0])
    c_pred = clr.LinearSegmentedColormap.from_list(
        'Blues', ['#A7BED3', '#114083', '#1A254B'], N=predicted.shape[0])

    for t in range(data.shape[0]):
        ax.scatter(data[t][:, 0], data[t][:, 1], c=[c_data(t)],
                   label='data, t={}'.format(t), marker='^')

    for t in range(predicted.shape[0]):
        ax.scatter(predicted[t][:, 0], predicted[t][:, 1], c=[c_pred(t)],
                   label='predicted, t={}'.format(t + 1))

    ax.legend(bbox_to_anchor=(0.5, 1.25), fontsize='medium',
              loc='upper center', ncol=3,
              columnspacing=1, frameon=False)

    fig.tight_layout()

    if save_loc is None:
        # plot in summary writer
        wandb.log({name + "predictions":
                   [wandb.Image(fig, caption="Predictions")]})
        plt.close('all')
    else:
        # save plot
        fig.savefig(save_loc + '.pdf')


def plot_potential_field(state, data, name='', save_loc=None):
    # set max and min values
    x_min = np.amin(data, axis=0)[:, 0].min() - 2.0
    x_max = np.amax(data, axis=0)[:, 0].max() + 2.0

    y_min = np.amin(data, axis=0)[:, 1].min() - 2.0
    y_max = np.amax(data, axis=0)[:, 1].max() + 2.0

    # create grid
    x, y = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))
    grid = np.vstack((x.ravel(), y.ravel())).T
    if data.shape[2] > 2:
        grid = np.concatenate((grid, np.zeros((grid.shape[0],
                                               data.shape[2]-2))), axis=1)
    # get predictions
    grad_pot_data = jax.vmap(lambda x: jax.grad(state.apply_fn, argnums=1)(
        state.params, x))(grid)
    predicted = grid + grad_pot_data

    # plot energy predictions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
    ax.grid(False)

    ax.quiver(grid[:, 0], grid[:, 1], predicted[:, 0], predicted[:, 1],
              color='k')

    fig.tight_layout()

    if save_loc is None:
        # plot in summary writer
        wandb.log({name + "potential_field":
                  [wandb.Image(fig, caption="Potential Field")]})
        plt.close('all')
    else:
        # save plot
        fig.savefig(save_loc + '.pdf')
