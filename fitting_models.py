import numpy as np
from scipy import stats
import scipy

import utils

SPLIT_SIZE = 8
NONE = -1
MY_METHOD = 0
LINEAR_LOG = 1
LOG_LINEAR = 2
LINEAR = 3
LOG = 4
name_of_type = {MY_METHOD: "my", LINEAR_LOG: "linear-log", LINEAR: "linear", LOG: "log", LOG_LINEAR: "log-linear"}


class Fitter:
    def __init__(self, x, y):
        mask = ~np.isnan(x) & ~np.isnan(y)
        self.x = x[mask]
        self.y = y[mask]
        self.method = None
        self.fitted_data = {}

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
                                       p0=(initital_a, 1, 0), maxfev=50000)
        self.fitted_data = {}
        self.fitted_data["a"] = res[0][0]
        self.fitted_data["b"] = res[0][1]
        self.fitted_data["c"] = res[0][2]
        self.method = LOG
        return self

    def fit_my(self):
        mask = ~np.isnan(self.x)
        x, y = self.x[mask], self.y[mask]
        splitted_x, splitted_y = utils.split_array(x,SPLIT_SIZE), utils.split_array(y,SPLIT_SIZE)
        self.fitted_data = {"acordionicity": [], "locations": []}
        for i in range(len(splitted_x)):
            accordionicity = (splitted_y[i][-1]-splitted_y[i][0])/(splitted_x[i][-1]-splitted_x[i][0])
            self.fitted_data["acordionicity"].append(np.abs(accordionicity))
            self.fitted_data["locations"].append(splitted_x[i][-1])
            self.fitted_data["start_y"] = self.y[mask][0]
        if y[-1] > y[0]:
            self.fitted_data["direction"] = 1
        else:
            self.fitted_data["direction"] = -1
        self.method = MY_METHOD
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
        elif self.method == MY_METHOD:
            new_x = utils.rescale_array(x, self.fitted_data["acordionicity"], self.fitted_data["locations"])
            return self.fitted_data["start_y"] + self.fitted_data["direction"] * new_x
        else:
            return None

    def score(self, x, y):
        """returns the mse"""
        mask = ~np.isnan(x) & ~np.isnan(y)
        if self.method is None:
            return None
        mask = mask & ~np.isnan(self.predict(x))
        predicted_y = self.predict(x[mask])
        return np.sqrt(np.average(np.power(y[mask] - predicted_y, 2)))

    def create_results_graph(self, x_test, y_test, cg_name="", type=LINEAR):
        y = self.y
        x = self.x

        if type == LINEAR:
            fitter = self.fit_linear()
        elif type == LOG:
            fitter = self.fit_log()
        elif type == MY_METHOD:
            fitter = self.fit_my()
        else:
            fitter = self.fit_linear()
        x_to_fit = np.linspace(np.nanmin(self.x), np.nanmax(self.x), 300)
        fitted = fitter.predict(x_to_fit)
        test_mask = ~np.isnan(x_test) & ~np.isnan(y_test)
        utils.plot_graph_nicely([x, x_to_fit, x_test], [y, fitted, y_test],
                                cg_name + " " + name_of_type[type] + ", loss = " + str(
                                    round(fitter.score(x_test[test_mask], y_test[test_mask]), 2)))
