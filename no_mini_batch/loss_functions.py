import numpy as np

class MSE:
    def f(self, y_true, y_pred):
        return np.sum(0.5*(y_true-y_pred)**2)
    def df(self, y_true, y_pred):
        return y_pred-y_true


class CEE:
    def f(self, y_true, y_pred):
        return - np.sum(y_true*np.log(y_pred))
    def df(self, y_true, y_pred):
        return np.where(y_pred != 0, - y_true / y_pred, 0)
