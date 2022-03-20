import torch
from SciDevNN.SDbase import Module


class BatchNorm2D(Module):
    def __init__(self, num_channels, eps, optimizer=None, init=None):
        self.eps = eps
        self.num_channels = num_channels

        self.optimizer = optimizer

        stdv = 1. / torch.sqrt(torch.tensor(num_channels))

        self.output = None
        self.input = None
        self.xhat = None
        self.X_var = None
        self.X_mean = None

        if init is None:
            self.W = torch.randn(num_channels).uniform_(-stdv, stdv)
            self.b = torch.randn(num_channels).uniform_(-stdv, stdv)
        else:
            self.W = init['W']
            self.b = init['b']

        self.dW = torch.zeros_like(self.W)
        self.db = torch.zeros_like(self.b)

        self.dx = None
        self.dxhat = None

    def forward(self, x):
        self.input = x
        m, c, h, w = x.shape
        N = (m * h * w)
        self.X_mean = torch.sum(x, dim=(0, 2, 3), keepdim=True)/N
        self.X_var = torch.sum((x-self.X_mean)**2, dim=(0, 2, 3), keepdim=True)/N

        self.xhat = (x - self.X_mean)/torch.sqrt(self.X_var + self.eps)
        self.output = self.W.view(1, self.num_channels, 1, 1) * self.xhat + self.b.view(1, self.num_channels, 1, 1)

        return self.output

    def backward(self, grad_output):

        m, c, h, w = self.input.shape
        Nt = (m * h * w)

        self.db = torch.sum(grad_output, dim=(0, 2, 3))
        self.dW = torch.sum(grad_output * self.xhat, dim=(0, 2, 3))

        self.dxhat = grad_output * self.W.reshape(1, self.num_channels, 1, 1)
        dsigma = torch.sum(self.dxhat * (self.input - self.X_mean), dim=(0, 2, 3)).view(1, self.num_channels, 1, 1) * -0.5 * (self.X_var + self.eps) ** -1.5
        dmu = torch.sum(self.dxhat * (-1.0 / torch.sqrt(self.X_var + self.eps)), dim=(0, 2, 3)).view(1, self.num_channels, 1, 1) + \
              dsigma * torch.sum(-2 * (self.input - self.X_mean), dim=(0, 2, 3)).view(1, self.num_channels, 1, 1) / Nt

        self.dx = self.dxhat * (1.0 / torch.sqrt(self.X_var + self.eps)) + dsigma * (2.0 * (self.input - self.X_mean))/Nt + dmu / Nt
        return self.dx

    def get_grad_test(self):
        return {'gradOut': self.dx, 'gradW': self.dW, 'gradB': self.db}

    def zero_grad(self):
        self.db *= 0
        self.dW *= 0

    def apply_grad(self):
        if self.optimizer is not None:
            self.optimizer.step(self.W, self.dW)
            self.optimizer.step(self.b, self.db)
        else:
            raise Exception("Нет оптимизатора, передайте оптимизатор в модель")
