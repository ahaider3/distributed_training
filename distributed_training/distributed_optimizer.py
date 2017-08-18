import tensorflow as tf
"""
Base class for distributed Optimizers
"""

class DistributedOptimizer:


    def __init__(self, opt, comm):
        self._opt = opt
        self._comm = comm
        self._num_workers = comm.num_workers


    def compute_gradients(self):
        pass



    def apply_gradients(self):
        pass
