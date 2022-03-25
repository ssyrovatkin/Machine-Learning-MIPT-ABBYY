import torch
from SciDevNN.SDbase import Module


class Softmax(Module):
    def __init__(self):
        super(Softmax, self).__init__()
        self.grad_input = None
        self.output = None

    def forward(self, x):
        self.output = (torch.exp(x))/torch.sum(torch.exp(x), dim=1, keepdim=True)
        return self.output

    def backward(self, x, grad_output):
        local_derivative = torch.zeros((grad_output.shape[0], grad_output.shape[1], grad_output.shape[1]))
        local_softmax = self.output

        for i in range(grad_output.shape[0]):
            local_derivative[i] = torch.diag(local_softmax[i])
            local_derivative[i] -= local_softmax[i].view(-1, 1) @ local_softmax[i].view(-1, 1).T

        self.grad_input = torch.matmul(grad_output[:, None, :], local_derivative)[:, 0, :]
        return self.grad_input

    def get_grad_test(self):
        return self.grad_input


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.grad_input = None
        self.output = None
        self.input = None

    def forward(self, x):
        self.input = x
        self.output = 1/(1+torch.exp(-x))
        return self.output

    def backward(self, grad_output):
        local = self.output*(1 - self.output)
        self.grad_input = local * grad_output
        return self.grad_input

    def get_grad_test(self):
        return self.grad_input


class ReLU(Module):
    def forward(self, *args):
        return torch.maximum(torch.FloatTensor(args), torch.FloatTensor(0))

    def backward(self, x, grad_output):
        gradInput = torch.multiply(grad_output, x > 0)
        return gradInput


class Tanh(Module):
    def forward(self, *args):
        args = torch.FloatTensor(args)
        return (torch.exp(args)-torch.exp(-args))/(torch.exp(args)+torch.exp(-args))

    def backward(self, x, grad_output):
        local = 4/(torch.exp(x)+torch.exp(-x))*(torch.exp(x)+torch.exp(-x))
        return local * grad_output
