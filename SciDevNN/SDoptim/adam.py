import torch
import numpy as np

from SciDevNN.SDbase import Optimizer

class Adam(Optimizer):
    def __init__(self, lr, betas):
        super(Adam, self).__init__()
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.m = 0
        self.v = 0
        self.t = 0
        self.eps = 1e-3
        self.current_lr = lr

        self.weights = None

    def step(self, weights, grad):
        self.t += 1
        self.current_lr = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        self.v = self.beta2*self.v + (1-self.beta2)*grad*grad

        self.m = self.beta1*self.m + (1-self.beta1)*grad

        self.weights = weights - self.current_lr*self.m/torch.sqrt(self.v + self.eps)

        return self.weights
