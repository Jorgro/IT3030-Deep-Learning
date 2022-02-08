import numpy as np

"""
This file contains the loss functions. All loss functions have f, and f_prime methods.
"""


class MSE:

    @staticmethod
    def f_prime(y_true, y_pred):
        return y_pred-y_true

    @staticmethod
    def f(y_true, y_pred):
        return np.sum(0.5*(y_true-y_pred)**2)


class CrossEntropy:

    @staticmethod
    def f(y_true, y_pred):
        return - np.sum(y_true*np.log(y_pred))

    @staticmethod
    def f_prime(y_true, y_pred):
        return np.where(y_pred != 0, - y_true / y_pred, 0)
