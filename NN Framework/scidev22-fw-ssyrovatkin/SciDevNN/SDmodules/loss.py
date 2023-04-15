import torch

from functools import reduce

from SciDevNN.SDbase import Module
from SciDevNN.SDmodules.activations import Softmax


class MSE(Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, x, y):
        raise NotImplementedError
    def backward(self, x, y):
        raise NotImplementedError

    def get_grad_test(self):
        return self.grad_input


class CrossEntropy(Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, x, y):
        raise NotImplementedError

    def backward(self, x, y):
        raise NotImplementedError

    def get_grad_test(self):
        return self.grad_input