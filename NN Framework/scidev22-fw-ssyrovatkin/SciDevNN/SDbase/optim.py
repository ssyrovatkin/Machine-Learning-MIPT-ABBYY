from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, *args):
        pass

    @abstractmethod
    def step(self, weights, grad):
        raise NotImplementedError
