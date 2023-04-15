import torch

from SciDevNN.SDbase import Optimizer


class Adam(Optimizer):
    def __init__(self, lr, betas):
        super(Adam, self).__init__()

    def step(self, weights, grad):
        raise NotImplementedError
