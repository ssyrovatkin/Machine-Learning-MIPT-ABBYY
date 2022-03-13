import torch
from SciDevNN.SDbase import Module


class Softmax(Module):
    def __init__(self):
        super(Softmax, self).__init__()
        self.grad_input = None
        self.output = None

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x, grad_output):
        raise NotImplementedError

    def get_grad_test(self):
        return self.grad_input


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x, grad_output):
        raise NotImplementedError

    def get_grad_test(self):
        return self.grad_input


class ReLU(Module):
    def forward(self, *args):
        raise NotImplementedError

    def backward(self, x, grad_output):
        raise NotImplementedError


class Tanh(Module):
    def forward(self, *args):
        raise NotImplementedError

    def backward(self, x, grad_output):
        raise NotImplementedError
