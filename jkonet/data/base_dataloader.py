#!/usr/bin/python3
# author: Charlotte Bunne


class BaseDynamicsDataset(object):
    """An abstract class for loading dynamics data via snapshots."""

    def __init__(self, name, batch_size):
        self.name = name
        self.batch_size = batch_size

    def __len__(self):
        pass

    def __getitem__(self, item):
        raise NotImplementedError
