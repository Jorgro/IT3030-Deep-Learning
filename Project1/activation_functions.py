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

    def __repr__(self):
        return "Sigmoid"


class ReLU(ActivationFunction):
    def f(self, x):
        return x * (x > 0)

    def df(self, x):
        return 1.0 * (x > 0)

    def __repr__(self):
        return "ReLU"


class Softmax(ActivationFunction):
    def f(self, x):
        exp_max = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_max / np.sum(exp_max, axis=1, keepdims=True)

    def df(self, x):
        return np.diagflat(x) + np.outer(x, -x)

    def __repr__(self):
        return "Softmax"


if __name__ == "__main__":
    relu = ReLU()
    print(relu.df(-0.01))
    arr = np.array([[2, 3, 6, 8], [3, 5, 4, 10]])
    softmax = Softmax()
    print(softmax.df(np.array([2, 3])))
    print(softmax.f(np.array([[2, 3, 6, 8], [3, 5, 4, 10]])))
    sigmoid = Sigmoid()
    print(sigmoid.f(np.array([[2, 3, 6, 8], [3, 5, 4, 10]])))

