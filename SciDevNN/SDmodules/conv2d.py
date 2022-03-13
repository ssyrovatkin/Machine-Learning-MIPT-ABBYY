import torch
import torch.nn.functional as F
import math
from SciDevNN.SDbase import Module
import copy

class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, init=None,
                 optimizer=None):
        pass

    def forward(self, X):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def get_grad_test(self):
        return {'gradOut': self.dout, 'gradW': self.dW, 'gradB': self.db}

    def zero_grad(self):
        pass

    def apply_grad(self):
        pass
