import torch
import math
from SciDevNN.SDbase import Module
import copy


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, init=None,
                 optimizer=None):
        super(Linear, self).__init__()

        pass

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x, grad_output):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    def apply_grad(self):
        raise NotImplementedError

    def get_grad_test(self):
        return {'gradW': self.gradW.T,
                'gradB': self.gradb,
                'gradOut': self.grad_input,
                }
