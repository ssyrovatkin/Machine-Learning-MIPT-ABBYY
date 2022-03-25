import torch
import torch.nn.functional as F
import math
from SciDevNN.SDbase import Module
import copy

class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, init=None,
                 optimizer=None):
        if init is None:
            stdv = 1. / math.sqrt(in_channels)
            self.W = torch.randn(out_channels, in_channels, kernel_size, kernel_size).uniform_(-stdv, stdv)
            self.b = torch.randn((out_channels,)).uniform_(-stdv, stdv)
        else:
            self.W = init['W']
            self.b = init['b']

        self.optimizerW = copy.deepcopy(optimizer)
        self.optimizerb = copy.deepcopy(optimizer)

        if self.optimizerW is None:
            raise Exception("Нет оптимизатора, передайте оптимизатор в модель")

        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.gradW = torch.zeros_like(self.W)
        self.gradb = torch.zeros_like(self.b)

        self.output = None
        self.input = None
        self.grad_input = None

    def forward(self, x):

        N, _, H, W = x.shape
        H = 1 + int((H + 2 * self.padding - self.kernel_size) // self.stride)
        W = 1 + int((W + 2 * self.padding - self.kernel_size) // self.stride)
        xpad = F.pad(x, (self.padding, self.padding, self.padding, self.padding), "constant", 0)
        self.input = xpad
        self.output = torch.zeros((N, self.out_channels, H, W))
        for xn in range(N):
            for fn in range(self.out_channels):
                for i in range(H):
                    h_start = i * self.stride
                    h_end = i * self.stride + self.kernel_size
                    for j in range(W):
                        w_start = j * self.stride
                        w_end = j * self.stride + self.kernel_size
                        self.output[xn, fn, i, j] = torch.sum(xpad[xn, :, h_start : h_end, w_start:w_end] * self.W[fn]) + self.b[fn]

        del xpad

        return self.output

    def backward(self, grad_output):
        N, Cout, Hout, Wout = grad_output.shape
        self.grad_input = torch.zeros(self.input.shape)
        self.gradb = torch.sum(grad_output, dim=(0, 2, 3))
        for xn in range(N):
            for fn in range(self.out_channels):
                for i in range(Hout):
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    for j in range(Wout):
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        self.grad_input[xn, :, h_start:h_end, w_start:w_end] += grad_output[xn, fn, i, j] * self.W[fn]
                        self.gradW[fn, :] += self.input[xn, :, h_start:h_end, w_start:w_end] * grad_output[xn, fn, i, j]

        if(self.padding):
            self.grad_input = self.grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return self.grad_input

    def get_grad_test(self):
        return {'gradOut': self.grad_input, 'gradW': self.gradW, 'gradB': self.gradb}

    def zero_grad(self):
        self.gradW *= 0
        self.gradb *= 0

    def apply_grad(self):
        self.W = self.optimizerW.step(self.W, self.gradW)
        self.b = self.optimizerb.step(self.b, self.gradb)


