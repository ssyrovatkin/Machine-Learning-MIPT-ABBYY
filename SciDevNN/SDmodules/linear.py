import torch
import math
from SciDevNN.SDbase import Module
import copy


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, init=None,
                 optimizer=None):
        super(Linear, self).__init__()

        self.grad_input = None
        self.output = None
        self.bias = bias

        stdv = 1. / torch.sqrt(torch.tensor(in_features))
        if init is None:
            self.W = torch.randn(in_features, out_features).uniform_(-stdv, stdv)
        else:
            self.W = init['W']
        self.gradW = torch.zeros_like(self.W)

        if bias:
            if init is None:
                self.b = torch.FloatTensor(out_features).uniform_(-stdv, stdv)
            else:
                self.b = init['b']
            self.gradb = torch.zeros_like(self.b)
        else:
            self.b = torch.zeros(out_features)
            self.gradb = torch.zeros_like(self.b)

    def forward(self, x):

        self.output = x @ self.W + self.b

        return self.output

    def backward(self, x, grad_output):
        self.grad_input = grad_output @ self.W.T

        self.gradW = x.T @ grad_output
        if self.bias:
            self.gradb = torch.sum(grad_output, dim=0)

        return self.grad_input

    def zero_grad(self):
        self.gradW = torch.zeros_like(self.gradW)
        if self.bias:
            self.gradb = torch.zeros_like(self.gradb)

    def apply_grad(self):
        if self.bias:
            return [self.gradW, self.gradb]
        else:
            return [self.gradW]

    def get_grad_test(self):
        return {'gradW': self.gradW.T,
                'gradB': self.gradb,
                'gradOut': self.grad_input,
                }
