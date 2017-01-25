import numpy as np
from numpy.linalg import inv


class Regression:
    def __init__(self):
        self.w = None
        self.b = None

    @property
    def weights(self):
        if self.w is None:
            raise Exception("Model not yet trained")
        return self.w

    @property
    def offset(self):
        if self.b is None:
            raise Exception("Model not yet trained")
        return self.b


class LinearRegression(Regression):
    def __init__(self):
        super().__init__()

    def fit(self, x, y):
        xt = np.insert(x, len(x[0]), values=1, axis=1)
        xtt = np.transpose(xt)
        model = ((inv(xtt.dot(xt))).dot(xtt)).dot(y)
        self.w = model[0:-1].flatten()
        self.b = model[-1].flatten()
        return self.w, self.b

    def predict(self, x):
        return np.array(list(map(lambda row: self.w.dot(row) + self.b, x)))


class NonlinearRegression(Regression):
    def __init__(self, fun):
        super().__init__()
        self.fun = fun

    def fit(self, x, y):
        x_transformed = self.fun(x)
        xt = np.insert(x_transformed, len(x_transformed[0]), values=1, axis=1)
        xtt = np.transpose(xt)
        model = ((inv(xtt.dot(xt))).dot(xtt)).dot(y)
        self.w = model[0:-1].flatten()
        self.b = model[-1].flatten()
        return self.w, self.b

    def predict(self, x):
        x_transformed = self.fun(x)
        return np.array(list(map(lambda row: self.w.dot(row) + self.b, x_transformed)))
