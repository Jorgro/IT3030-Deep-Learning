import abc
import numpy as np


class ActivationFunction(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def f(self, x):
        pass

    @abc.abstractclassmethod
    def df(self, x):
        pass


class Sigmoid(ActivationFunction):
    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def df(self, x):
        return self.f(x) * (1 - self.f(x))

    def __repr__(self):
        return "Sigmoid"


class ReLU(ActivationFunction):
    def f(self, x):
        return x * (x > 0)

    def df(self, x):
        return 1.0 * (x > 0)

    def __repr__(self):
        return "ReLU"


class LeakyReLU(ActivationFunction):
    def f(self, x):
        return np.where(x > 0, x, x * 0.1)

    def df(self, x):
        return np.where(x > 0, 1.0, 0.1)

    def __repr__(self):
        return "LeakyReLU"


# CHANGE!
class Linear(ActivationFunction):
    def f(self, x):
        return x

    def df(self, _):
        return 1

    def __repr__(self):
        return "Linear"


class Tanh(ActivationFunction):
    def f(self, x):
        return np.tanh(x)

    def df(self, x):
        return 1 - self.f(x) ** 2

    def __repr__(self):
        return "Tanh"


class Softmax(ActivationFunction):
    def f(self, x):
        exp_max = np.exp(x - np.max(x))
        return exp_max / np.sum(exp_max)

    def df(self, x):
        return np.diag(x) - np.outer(x, x)

    def __repr__(self):
        return "Softmax"


class MSE:
    def f(self, y, y_p):
        return 1 / 2 * np.sum((y_p - y) ** 2)

    def df(self, y, y_p):
        return y_p - y


class CEE:
    def f(self, y, y_p):
        return - np.sum(y*np.log(y_p))

    def df(self, y, y_p):
        return  np.where(y_p != 0, - y / y_p, 0)


