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


# CHANGE!
class Tanh(ActivationFunction):
    def f(self, x):
        return np.tanh(x)

    def df(self, x):
        return 1 - self.f(x) ** 2

    def __repr__(self):
        return "Tanh"


class Softmax(ActivationFunction):
    def f(self, x):
        exp_max = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_max / np.sum(exp_max, axis=1, keepdims=True)

    def df(self, x):
        _, n = x.shape
        x = self.f(x)
        return np.einsum("ij,jk->ijk", x, np.eye(n, n)) - np.einsum("ij,ik->ijk", x, x)

    def __repr__(self):
        return "Softmax"


if __name__ == "__main__":
    relu = LeakyReLU()
    print(relu.df(-0.01))
    """  arr = np.array([[2, 3, 6, 8], [3, 5, 4, 10]])
    softmax = Softmax()
    print(softmax.df(np.array([2, 3])))
    print(softmax.f(np.array([[2, 3, 6, 8], [3, 5, 4, 10]])))
    sigmoid = Sigmoid()
    print(sigmoid.f(np.array([[2, 3, 6, 8], [3, 5, 4, 10]])))

    a = np.array([np.arange(1, 5)])
    y = np.array([[0, 1, 0, 0]])
    s = softmax.f(a)
    print("s: ", s)
    c = -y / s
    print("c: ", c)
    print("c @ s': ", c @ softmax.df(s))
    print("s - y: ", s - y)
    print("Grad: ", c @ softmax.gradient(s)) """

