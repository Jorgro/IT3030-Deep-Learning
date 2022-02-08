import numpy as np


class Sigmoid:
    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def df(self, x):
        return self.f(x) * (1 - self.f(x))

    def __repr__(self):
        return "Sigmoid"


class Softmax:
    def f(self, x):
        exp_max = np.exp(x - np.max(x))
        return exp_max / np.sum(exp_max)

    def df(self, x):
        return np.diag(x) - np.outer(x, x)

    def __str__(self) -> str:
        return "Softmax"

    def __repr__(self):
        return "Softmax"


class ReLU:
    def f(self, x):
        return x * (x > 0)

    def df(self, x):
        return 1.0 * (x > 0)

    def __repr__(self):
        return "ReLU"


class Tanh:
    def f(self, x):
        return np.tanh(x)

    def df(self, x):
        return 1 - self.f(x) ** 2

    def __repr__(self):
        return "Tanh"


class Linear:
    def f(self, x):
        return x

    def df(self, x):
        return np.ones(x.shape)
