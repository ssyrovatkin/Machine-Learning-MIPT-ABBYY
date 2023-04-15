from SciDevNN.SDbase import Optimizer


class GradientDescend(Optimizer):
    def __init__(self, lr):
        super(GradientDescend, self).__init__()

    def step(self, weights, grad):
        raise NotImplementedError
