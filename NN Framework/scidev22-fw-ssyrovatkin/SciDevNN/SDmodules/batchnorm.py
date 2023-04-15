import torch
from SciDevNN.SDbase import Module


class BatchNorm2D(Module):
    def __init__(self, num_channels, eps, optimizer=None, init=None):
        pass

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def get_grad_test(self):
        return {'gradOut': self.dx, 'gradW': self.dW, 'gradB': self.db}

    def zero_grad(self):
        pass

    def apply_grad(self):
        pass
