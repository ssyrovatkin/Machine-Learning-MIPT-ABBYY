from SciDevNN.SDbase import Module


class RNNLayer(Module):
    def __init__(self, output_size, cell_type=None, loss=None, optimizer=None):
        super(RNNLayer, self).__init__()
        raise NotImplementedError 

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x, y):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    def apply_grad(self):
        raise NotImplementedError

    def train(self, data, n_epochs):
        raise NotImplementedError