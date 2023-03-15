import numpy as np
from scipy import stats
import scipy

import utils

NONE = -1
MY_METHOD = 0
COMMON = 1
LINEAR = 2
LOG = 3
name_of_type = {MY_METHOD: "my", COMMON: "common", LINEAR: "linear", LOG: "log"}


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
        res = scipy.optimize.curve_fit(lambda t, a, b, c: a * np.log(b * (t + c)), self.x[mask], self.y[mask],
                                       p0=(initital_a, 1, 0), maxfev=5000)
        self.fitted_data = {}
        self.fitted_data["a"] = res[0][0]
        self.fitted_data["b"] = res[0][1]
        self.fitted_data["c"] = res[0][2]
        self.method = LOG
        return self

    def predict(self, x=None):
        if x is None:
            x = self.x
        if self.method is None:
            return None
        elif self.method == LINEAR:
            return self.fitted_data["gradient"] * x + self.fitted_data["intercept"]
        elif self.method == LOG:
            return self.fitted_data["a"] * np.log(self.fitted_data["b"] * (x + self.fitted_data["c"]))
        else:
            return None

    def score(self, x, y):
        """returns the mse"""
        if self.method is None:
            return None
        if self.method == LOG:
            mask = ~np.isnan(x) & ~np.isnan(y) & (self.fitted_data["b"]*(x + self.fitted_data["c"]) > 0)
        else:
            mask = ~np.isnan(x) & ~np.isnan(y)
        predicted_y = self.predict(x[mask])
        if np.nan in predicted_y:
            print(1)
        return np.sqrt(np.average(np.power(y[mask] - predicted_y, 2)))

    def create_results_graph(self, x_test, y_test, cg_name="", type=LINEAR):
        y = self.y
        x = self.x

        if type == LINEAR:
            fitter = self.fit_linear()
        else:
            fitter = self.fit_log()
        fitted = fitter.predict(x)
        utils.plot_graph_nicely([x, x, x_test], [y, fitted, y_test],
                                cg_name + " " + name_of_type[type] + ", loss = " + str(
                                    round(fitter.score(x_test, y_test), 2)))
