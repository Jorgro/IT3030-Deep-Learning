import abc
import numpy as np


class ActivationFunction(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def f(self, x):
        pass

    @staticmethod
    def df(self, x):
        pass


class Sigmoid(ActivationFunction):
    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def df(self, x):
        return self.f(x) * (1 - self.f(x))

