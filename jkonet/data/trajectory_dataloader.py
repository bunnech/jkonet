#!/usr/bin/python3
# author: Charlotte Bunne

# imports
from itertools import chain
import numpy as np
from jkonet.data.base_dataloader import BaseDynamicsDataset


class TrajectoryDynamicsDataset(BaseDynamicsDataset):
    """Dataset class for loading snapshot dynamics data from trajectories."""

    def __init__(self, name, batch_size, time_steps, missing_values=None, noise=None):
        super(TrajectoryDynamicsDataset, self).__init__(name, batch_size)
        self.time_steps = time_steps
        self.dim = 2
        self.trajectory = self.get_trajectory()
        self.missing_values = missing_values
        self.noise = noise

    def __len__(self):
        pass

    def __getitem__(self, idx):
        return next(self.train())

    def train(self):
        return self.generate_dataset()

    def evaln(self):
        return self.generate_dataset(evaln=True)

    def testn(self):
        return self.generate_dataset(testn=True)

    def generate_dataset(self, scale=1.0, evaln=False, testn=False):
        """Generate dataset from trajectory."""
        if testn and self.name == "line":
            self.trajectory = self.long_line_trajectory()

        # split trajectory by number of time steps
        ts = np.array_split(self.trajectory, self.time_steps)

        # subsample observation
        while True:
            data_sample = []
            for t in ts:
                data = np.array([]).reshape(0, 2)
                quotient, remainder = divmod(self.batch_size, len(t))
                ns = [quotient + int(i < remainder) for i in range(len(t))]

                for i, n in enumerate(ns):
                    sample = t[i] + np.random.randn(n, 2) * scale
                    data = np.concatenate((data, sample))
                data_sample.append(data)
            data_sample = np.array(data_sample, dtype="float32")

            # add outliers if noise level is specified
            if self.missing_values and not evaln and not testn:
                mv_mask = (
                    np.random.uniform(size=(self.time_steps - 1, self.batch_size, 2))
                    < self.missing_values
                ).astype(int)
                # data_sample += mv_mask * np.random.randn(
                # self.time_steps, self.batch_size, 2) * self.noise
                data_sample[1:, :, :] += mv_mask * self.noise
            yield data_sample

    def get_trajectory(self):
        if self.name == "circle":
            return self.circle_trajectory()
        elif self.name == "semicircle":
            return self.semicircle_trajectory()
        elif self.name == "tee":
            return self.tee_trajectory()
        elif self.name == "spiral":
            return self.spiral_trajectory()
        elif self.name == "tree":
            return self.tree_trajectory()
        elif self.name == "line":
            return self.line_trajectory()
        else:
            raise NotImplementedError

    def circle_trajectory(self):
        # theta between 0 and 2 pi
        theta = np.linspace(0, 2 * np.pi, 100)

        # circle radius
        r = 1

        # create circle trajectory
        x = np.flip(r * np.cos(theta) * 10)
        y = np.flip(r * np.sin(theta) * 10)
        return np.stack([x, y], axis=1)

    def semicircle_trajectory(self):
        # theta between 0 and 2 pi
        theta = np.linspace(2 * np.pi, 0, 100)

        # circle radius
        r = 10

        # create circle trajectory
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # create semicircle trajectory
        x = x[np.where(y >= 0)[0]]
        y = y[np.where(y >= 0)[0]]
        return np.stack([x, y], axis=1)

    def spiral_trajectory(self):
        # theta between 0 and 3 pi
        theta = np.linspace(2.75 * np.pi, 0, 150)

        # circle radius
        r = np.linspace(15, 1, 150)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        x = np.concatenate((x[:50], x[50::2]))
        y = np.concatenate((y[:50], y[50::2]))
        return np.stack([x, y], axis=1)

    def line_trajectory(self):
        # create line trajectory
        x = np.linspace(-10, -2.5, 100)
        y = np.zeros(100)

        return np.stack([x, y], axis=1)

    def long_line_trajectory(self):
        # create line trajectory
        x = np.linspace(-5, 7.5, 100)
        y = np.zeros(100)

        return np.stack([x, y], axis=1)

    def tree_trajectory(self):
        # create tree trajectory
        x0 = np.linspace(-10, -2.5, 100)
        x1 = np.linspace(-2.5, 15, 175)

        y0 = np.zeros(100)
        y1 = 0.75 * x1
        y2 = -0.75 * x1

        # create tree branches
        x = np.concatenate((x0, x1, x1))
        y = np.concatenate((y0, y1, y2)) + 10

        sidx = np.argsort(x)
        return np.stack([x[sidx], y[sidx]], axis=1)

    def tee_trajectory(self):
        # create tee trajectory
        x0 = np.linspace(0, 10, 11)
        y0 = x0 * 0
        y1 = np.linspace(0, 10, 11)
        x1 = y1 * 0
        y2 = np.linspace(0, -10, 11)
        x2 = y2 * 0

        t0 = np.stack([x0, y0], axis=1)
        t1 = np.stack([x1, y1], axis=1)
        t2 = np.stack([x2, y2], axis=1)

        # combine tee arms
        return np.stack(list(chain.from_iterable(zip(t0, t1, t2))))
