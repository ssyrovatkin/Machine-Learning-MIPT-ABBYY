import torch
import math
import copy

from SciDevNN.SDbase import Module
from SciDevNN.SDmodules.activations import Tanh, Sigmoid


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, init=None, optimizer=None, bias=True, batch_first=True):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.output = None

        self.optimizerWh = copy.deepcopy(optimizer)
        self.optimizerWi = copy.deepcopy(optimizer)
        self.optimizerbh = copy.deepcopy(optimizer)
        self.optimizerbi = copy.deepcopy(optimizer)

        self.batch_first = batch_first

        stdv = 1. / torch.sqrt(torch.tensor(hidden_size))

        if init is not None:
            self.Wh = init["W_hh"]
            self.Wi = init["W_ih"]
            if self.bias:
                self.bh = init["bh"]
                self.bi = init["bi"]
            else:
                self.bh = torch.zeros(self.hidden_size)
                self.bi = torch.zeros(self.hidden_size)
        else:
            self.Wh = torch.randn(self.hidden_size, self.hidden_size).uniform_(-stdv, stdv)
            self.Wi = torch.randn(self.hidden_size, self.input_size).uniform_(-stdv, stdv)
            if self.bias:
                self.bh = torch.randn(self.hidden_size).uniform_(-stdv, stdv)
                self.bi = torch.randn(self.hidden_size).uniform_(-stdv, stdv)
            else:
                self.bh = torch.zeros(self.hidden_size)
                self.bi = torch.zeros(self.hidden_size)

        self.gradWh = torch.zeros_like(self.Wh)
        self.gradWi = torch.zeros_like(self.Wi)
        self.gradbh = torch.zeros_like(self.bh)
        self.gradbi = torch.zeros_like(self.bi)

        self.grad_input_t = None
        self.grad_input_h = None
        self.input = None
        self.h = None

    def forward(self, x, h):

        scidev_tanh = Tanh()
        self.output = scidev_tanh(h @ self.Wh.T + self.bh + x @ self.Wi.T + self.bi)
        self.input = x

        self.h = h

        return self.output

    def backward(self, x, grad_output):

        local = (1 - self.output ** 2)
        self.grad_input_t = grad_output * local @ self.Wi
        self.grad_input_h = grad_output * local @ self.Wh

        self.gradWh = (grad_output * local).T @ self.h
        self.gradWi = (grad_output * local).T @ self.input
        if self.bias:
            self.gradbh = torch.sum(grad_output*local, dim=0)
            self.gradbi = torch.sum(grad_output*local, dim=0)

        return self.grad_input_t, self.grad_input_h

    def zero_grad(self):
        self.gradWh *= 0
        self.gradWi *= 0
        if self.bias:
            self.gradbh *= 0
            self.gradbi *= 0

    def apply_grad(self):
        if self.optimizerWh is None:
            raise Exception("Нет оптимизатора, передайте оптимизатор в модель")
        self.Wh = self.optimizerWh.step(self.Wh, self.gradWh)
        self.Wi = self.optimizerWh.step(self.Wi, self.gradWi)
        if self.bias:
            self.bh = self.optimizerbh.step(self.bh, self.gradbh)
            self.bi = self.optimizerbi.step(self.bi, self.gradbi)

    def get_grad_test(self):
        return {'gradWh': self.gradWh,
                'gradWi': self.gradWi,
                'gradbh': self.gradbh,
                'gradbi': self.gradbi,
                'gradx': self.grad_input_t,
                'gradh': self.grad_input_h
                }


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, init=None, optimizer=None, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.output = None

        self.optimizerWh = copy.deepcopy(optimizer)
        self.optimizerWi = copy.deepcopy(optimizer)
        self.optimizerbh = copy.deepcopy(optimizer)
        self.optimizerbi = copy.deepcopy(optimizer)

        stdv = 1. / torch.sqrt(torch.tensor(hidden_size))

        if init is not None:
            self.Wh = init["W_hh"]
            self.Wi = init["W_ih"]
            if self.bias:
                self.bh = init["bh"]
                self.bi = init["bi"]
            else:
                self.bh = torch.zeros(4*self.hidden_size)
                self.bi = torch.zeros(4*self.hidden_size)
        else:
            self.Wh = torch.randn(4*self.hidden_size, self.hidden_size).uniform_(-stdv, stdv)
            self.Wi = torch.randn(4*self.hidden_size, self.input_size).uniform_(-stdv, stdv)
            if self.bias:
                self.bh = torch.randn(4*self.hidden_size).uniform_(-stdv, stdv)
                self.bi = torch.randn(4*self.hidden_size).uniform_(-stdv, stdv)
            else:
                self.bh = torch.zeros(4*self.hidden_size)
                self.bi = torch.zeros(4*self.hidden_size)

        self.gradWh = torch.zeros_like(self.Wh)
        self.gradWi = torch.zeros_like(self.Wi)
        self.gradbh = torch.zeros_like(self.bh)
        self.gradbi = torch.zeros_like(self.bi)

        self.grad_input = None
        self.grad_c = None
        self.grad_h = None
        self.input = None
        self.h = None
        self.c = None
        self.h_prev = None
        self.c_prev = None
        self.logits = None

        self.i = None
        self.f = None
        self.g = None
        self.o = None

    def forward(self, x, h_c):

        self.h_prev = h_c[0]
        self.c_prev = h_c[1]
        self.input = x
        scidev_tanh = Tanh()
        scidev_sigm = Sigmoid()
        self.logits = torch.tensor_split(h_c[0] @ self.Wh.T + self.bh + x @ self.Wi.T + self.bi, 4, dim=1)

        self.i = scidev_sigm(self.logits[0])
        self.f = scidev_sigm(self.logits[1])
        self.g = scidev_tanh(self.logits[2])
        self.o = scidev_sigm(self.logits[3])

        self.c = self.f*h_c[1] + self.i*self.g
        self.h = self.o*scidev_tanh(self.c)

        return self.h, self.c

    def backward(self, x, grad_output):

        dWio = torch.zeros((self.input_size, self.hidden_size))
        dWho = torch.zeros((self.hidden_size, self.hidden_size))
        dbo = torch.zeros(self.hidden_size)

        dc = grad_output

        dg = dc * self.i
        da_g = dg * (1 - self.g ** 2)
        dWig = x.T @ da_g
        dWhg = self.h_prev.T @ da_g
        dbg = torch.sum(da_g, dim=0)

        di = dc * self.g
        da_i = di * self.i * (1 - self.i)
        dWii = x.T @ da_i
        dWhi = self.h_prev.T @ da_i
        dbi = torch.sum(da_i, dim=0)

        df = dc * self.c_prev
        da_f = df * self.f * (1 - self.f)
        dWif = x.T @ da_f
        dWhf = self.h_prev.T @ da_f
        dbf = torch.sum(da_f, dim=0)

        weights_i = torch.tensor_split(self.Wi, 4, dim=0)
        weights_h = torch.tensor_split(self.Wh, 4, dim=0)
        self.gradWi = torch.concat((dWii, dWif, dWig, dWio), 1).T
        self.gradWh = torch.concat((dWhi, dWhf, dWhg, dWho), 1).T
        self.gradbi = torch.concat((dbi, dbf, dbg, dbo))
        self.gradbh = torch.concat((dbi, dbf, dbg, dbo))

        self.grad_input = (da_i @ weights_i[0] + da_f @ weights_i[1] + da_g @ weights_i[2])
        self.grad_h = (da_i @ weights_h[0] + da_f @ weights_h[1] + da_g @ weights_h[2])
        self.grad_c = self.f * dc
        return self.grad_input

    def zero_grad(self):
        self.gradWh *= 0
        self.gradWi *= 0
        if self.bias:
            self.gradbh *= 0
            self.gradbi *= 0

    def apply_grad(self):
        if self.optimizerWh is None:
            raise Exception("Нет оптимизатора, передайте оптимизатор в модель")
        self.Wh = self.optimizerWh.step(self.Wh, self.gradWh)
        self.Wi = self.optimizerWh.step(self.Wi, self.gradWi)
        if self.bias:
            self.bh = self.optimizerbh.step(self.bh, self.gradbh)
            self.bi = self.optimizerbi.step(self.bi, self.gradbi)

    def get_grad_test(self):
        return {'gradWh': self.gradWh,
                'gradWi': self.gradWi,
                'gradbh': self.gradbh,
                'gradbi': self.gradbi,
                'gradx': self.grad_input,
                'gradh': self.grad_h,
                'gradc': self.grad_c,
                }


class GRUCell(Module):
    def __init__(self, input_size, hidden_size, init=None, optimizer=None):
        super(GRUCell, self).__init__()
        raise NotImplementedError

    def forward(self, x, h):
        raise NotImplementedError

    def backward(self, x, grad_output):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    def apply_grad(self):
        raise NotImplementedError