from abc import ABC, abstractmethod
from SciDevNN.SDbase import Module

class Model(Module, ABC):
    def __init__(self, *args, loss=None, optimizer=None):
        super(Model, self).__init__()
        self.loss = loss
        self.optimizer = optimizer

    def step(self, x, y):
        self.zero_grad()
        loss = self.backward(x, y)
        self.apply_grad()
        return loss

    def zero_grad(self):
        raise NotImplementedError
