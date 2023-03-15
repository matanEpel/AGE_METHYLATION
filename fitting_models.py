import numpy as np

NONE = -1
MY_METHOD = 0
COMMON = 1
LINEAR = 2
LOGARITHMIC = 3


class fitter:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.method = None

    def fit_my(self):
        pass

    def fit_common(self):
        pass

    def fit_linear(self):
        pass

    def fit_logarithm(self):
        pass

    def predict(self, x):
        if self.method is None:
            return None

    def score(self, x, y):
        """returns the average squared error"""
        if self.method is None:
            return None
        predicted_y = self.predict(x)
        return np.average(np.power(y-predicted_y, 2))
