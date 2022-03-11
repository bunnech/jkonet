#!/usr/bin/python3
# author: Charlotte Bunne

# imports
import numpy as np
from jkonet.data.base_dataloader import BaseDynamicsDataset


class PotentialDynamicsDataset(BaseDynamicsDataset):
    """Dataset class for loading snapshot dynamics data from potentials."""

    def __init__(self, name, batch_size, setting='single'):
        super(PotentialDynamicsDataset, self).__init__(name, batch_size)
        self.setting = setting
        self.data = {
            'quadratic': [
                self.quadratic_potential,
                [np.random.multivariate_normal([0, 0], [[-50, 0], [0, 50]],
                                               self.batch_size),
                 np.random.multivariate_normal([0, 40], [[1, 0], [0, 1]],
                                               self.batch_size)],
                {'dt': 0.25, 't': 1.0, 'sd': 0.2}],
            'styblinski': [
                self.styblinski_potential,
                [np.random.multivariate_normal([0, 0], [[.5, 0], [0, .5]],
                                               self.batch_size),
                 np.random.multivariate_normal([1, 1], [[.5, 0], [0, .5]],
                                               self.batch_size)],
                {'dt': 0.06, 't': 0.4, 'sd': 0.5}],
            'relu': [
                self.relu_potential,
                [np.random.multivariate_normal([40, 10], [[1, 0], [0, 1]],
                                               self.batch_size),
                 np.random.multivariate_normal([10, 40], [[1, 0], [0, 1]],
                                               self.batch_size),
                 np.random.multivariate_normal([30, 30], [[1, 0], [0, 1]],
                                               self.batch_size)],
                {'dt': 8, 't': 48, 'sd': 0.1}]
            }

    def __len__(self):
        pass

    def __getitem__(self, item):
        return next(self.train())

    def train(self):
        return self.generate_data()

    def evaln(self):
        return self.generate_data()

    def testn(self):
        return self.generate_data(test=True)

    def generate_data(self, test=False):
        """Generate dataset from potential."""
        while True:
            # get initial distribution
            potential, init, setting = self.get_potential_init(test)

            # simulate trajectory via Euler-Maruyama
            trajectory = self.euler_maruyama_method(
                init, potential, setting['dt'], setting['t'], setting['sd'])

            yield trajectory

    def euler_maruyama_method(self, p, flow, dt, t, sd):
        """Simulation of SDE via Euler-Maruyama method."""
        pp = np.copy(p)
        n = int(t / dt)
        pset = np.zeros((n, pp.shape[0], pp.shape[1]))
        sqrtdt = np.sqrt(dt)
        for i in range(n):
            drift = flow(pp)
            pp = pp + drift * dt + np.random.normal(scale=sd,
                                                    size=p.shape) * sqrtdt
            pset[i, :, :] = pp
        return pset

    def get_potential_init(self, test=False):
        # get potential function and iniit distributions
        potential, init_pops, setting = self.data[self.name]

        # depending on setting choose init distribution
        if self.setting == 'single':
            init = 0
        elif self.setting == 'multi':
            if test:
                init = -1
            else:
                init = np.random.randint(len(init_pops)-1)
        else:
            raise NotImplementedError

        return potential, init_pops[init], setting

    def styblinski_potential(self, v):
        return -1*(3*v**3-32*v+5)/2.0

    def quadratic_potential(self, v):
        return -2*v

    def relu_potential(self, v):
        def ilogit(x):
            return 1/(1+np.exp(-x))

        return -1*ilogit(v)

    def dipole_potential(self, v):
        pass
