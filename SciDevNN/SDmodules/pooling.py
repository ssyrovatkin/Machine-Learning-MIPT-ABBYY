from SciDevNN.SDbase import Module

class MaxPooling2D(Module):
    def __init__(self, input_size, hidden_size, init=None, optimizer=None):
        super(MaxPooling2D, self).__init__()
        raise NotImplementedError

    def forward(self, x, h):
        raise NotImplementedError

    def backward(self, x, grad_output):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    def apply_grad(self):
        raise NotImplementedError
