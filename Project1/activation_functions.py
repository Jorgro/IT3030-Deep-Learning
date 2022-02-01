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


class ReLU(ActivationFunction):
    def f(self, x):
        return x * (x > 0)

    def df(self, x):
        return 1.0 * (x > 0)

class Softmax(ActivationFunction):
    def f(self, x):
        e_x = np.exp(x -np.max(x))
        return e_x / e_x.sum(axis=0)

    def df(self, x):
        pass
