from SciDevNN.SDbase import Optimizer


class GradientDescend(Optimizer):
    def __init__(self, lr):
        super(GradientDescend, self).__init__()
        self.lr = lr
        self.weights = None

    def step(self, weights, grad):
        self.weights = weights - self.lr*grad
        return self.weights