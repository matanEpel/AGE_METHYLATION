import numpy as np
from scipy import stats
import scipy

NONE = -1
MY_METHOD = 0
COMMON = 1
LINEAR = 2
LOG = 3


class Fitter:
    def __init__(self, x, y):
        mask = ~np.isnan(x) & ~np.isnan(y)
        self.x = x[mask]
        self.y = y[mask]
        self.method = None
        self.fitted_data = {}

    def fit_my(self):
        return self

    def fit_common(self):
        return self

    def fit_linear(self):
        self.fitted_data = {}
        gradient, intercept, r_value, p_value, slope_std_error = stats.linregress(self.x, self.y)
        self.fitted_data["gradient"] = gradient
        self.fitted_data["intercept"] = intercept
        self.method = LINEAR
        return self

    def fit_log(self):
        """ a*exp(b*y)+c """
        mask = (self.x != 0)
        initital_a = -10
        if self.y[-1] > self.y[0]:
            initital_a = 10
        res = scipy.optimize.curve_fit(lambda t, a, b: a * np.log(b*t), self.x[mask], self.y[mask],  p0=(initital_a, 1), maxfev=5000)
        self.fitted_data = {}
        self.fitted_data["a"] = res[0][0]
        self.fitted_data["b"] = res[0][1]
        self.method = LOG
        return self

    def predict(self, x=None):
        if x is None:
            x = self.x
        if self.method is None:
            return None
        elif self.method == LINEAR:
            return self.fitted_data["gradient"]*x+self.fitted_data["intercept"]
        elif self.method == LOG:
            return self.fitted_data["a"]*np.log(self.fitted_data["b"]*x)
        else:
            return x  # TODO

    def score(self, x, y):
        """returns the mse"""
        if self.method is None:
            return None
        if self.method == LOG:
            mask = ~np.isnan(x) & ~np.isnan(y) & (x != 0)
        else:
            mask = ~np.isnan(x) & ~np.isnan(y)
        predicted_y = self.predict(x)
        return np.sqrt(np.average(np.power(y[mask]-predicted_y[mask], 2)))
