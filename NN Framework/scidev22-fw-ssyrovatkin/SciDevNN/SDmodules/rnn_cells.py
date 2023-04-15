from SciDevNN.SDbase import Module


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, init=None, optimizer=None):
        super(RNNCell, self).__init__()
        raise NotImplementedError

    def forward(self, x, h):
        raise NotImplementedError

    def backward(self, x, grad_output):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    def apply_grad(self):
        raise NotImplementedError


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, init=None, optimizer=None):
        super(LSTMCell, self).__init__()
        raise NotImplementedError

    def forward(self, x, h_c):
        raise NotImplementedError

    def backward(self, x, grad_output):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    def apply_grad(self):
        raise NotImplementedError


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