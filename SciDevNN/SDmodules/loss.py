import torch
from functools import reduce

from SciDevNN.SDbase import Module
from SciDevNN.SDmodules.activations import Softmax


class MSE(Module):
    def __init__(self):
        super(MSE, self).__init__()

        self.grad_input = None
        self.output = None

    def forward(self, x, y):
        self.output = torch.mean((x-y)*(x-y))
        return self.output
    def backward(self, x, y):
        self.grad_input = 2*(x-y)/y.numel()
        return self.grad_input

    def get_grad_test(self):
        return self.grad_input


class CrossEntropy(Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.grad_input = None
        self.output = None

    def forward(self, x, y):
        p = Softmax().forward(x)
        m = y.shape[0]

        log_likelihood = -torch.log(p[range(m), y])
        self.output = torch.sum(log_likelihood) / m
        return self.output


    def backward(self, x, y):
        self.grad_input = Softmax().forward(x)
        self.grad_input[range(y.shape[0]), y] -= 1
        self.grad_input /= y.shape[0]
        return self.grad_input

    def get_grad_test(self):
        return self.grad_input