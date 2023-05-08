import numpy as np
from scipy import stats
from my_fitter import MY_FITTER
import scipy

import utils

NONE = -1
MY_METHOD = 0
LINEAR = 3
LOG = 4
MAGE = 5
name_of_type = {MY_METHOD: "my", LINEAR: "linear", LOG: "log", MAGE: "mAge"}


class Fitter:
    def __init__(self, x, y):
        mask = ~np.isnan(x) & ~np.isnan(y)
        self.x = x[mask]
        self.y = y[mask]
        self.method = None
        self.fitted_data = {}
        self.my_fitter = MY_FITTER(x, y)

    def create_age_aacordionicity_graph(self):
        self.my_fitter.create_age_aacordionicity_graph()

    def fit_linear(self, to_20=False):
        mask_20 = (self.x > 20)
        if to_20:
            gradient, intercept, r_value, p_value, slope_std_error = stats.linregress(self.x[mask_20], self.y[mask_20])
        else:
            gradient, intercept, r_value, p_value, slope_std_error = stats.linregress(self.x, self.y)
        self.fitted_data["gradient"] = gradient
        self.fitted_data["intercept"] = intercept
        self.method = LINEAR
        return self

    def fit_log(self, to_20=False):
        """ a*exp(b*y)+c """
        mask = (self.x != 0)
        mask_20 = (self.x <= 20)
        initital_a = -10
        if self.y[-1] > self.y[0]:
            initital_a = 10
        if to_20:
            res = scipy.optimize.curve_fit(lambda t, a, b, c: a * np.log(b * (t + c)), self.x[mask & mask_20],
                                           self.y[mask & mask_20],
                                           p0=(initital_a, 1, 0), maxfev=500000)
        else:
            res = scipy.optimize.curve_fit(lambda t, a, b, c: a * np.log(b * (t + c)), self.x[mask], self.y[mask],
                                           p0=(initital_a, 1, 0), maxfev=500000)
        self.fitted_data["a"] = res[0][0]
        self.fitted_data["b"] = res[0][1]
        self.fitted_data["c"] = res[0][2]
        self.method = LOG
        return self

    def fit_mAge(self):
        self.fit_log(to_20=True)
        self.fit_linear(to_20=True)
        self.method = MAGE
        return self

    def fit_my(self, optimal=False, v=3):
        self.my_fitter.fit(optimal=optimal, v=v)
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
        elif self.method == MAGE:
            mask_to = (x <= 20)
            mask_from = (x > 20)
            first_part = self.fitted_data["a"] * np.log(self.fitted_data["b"] * (x[mask_to] + self.fitted_data["c"]))
            second_part = self.fitted_data["gradient"] * x[mask_from] + self.fitted_data["intercept"]
            return np.concatenate([first_part, second_part])
        elif self.method == MY_METHOD:
            return self.my_fitter.predict(x)

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

    def get_type(self):
        return self.my_fitter.get_type()

    def get_vector(self):
        return self.my_fitter.get_vector()

    def create_results_graph(self, x_test, y_test, cg_name="", type=LINEAR, optimal=False, v=3):
        y = self.y
        x = self.x

        if type == LINEAR:
            fitter = self.fit_linear()
        elif type == LOG:
            fitter = self.fit_log()
        elif type == MAGE:
            fitter = self.fit_mAge()
        elif type == MY_METHOD:
            self.my_fitter.create_results_graph(x_test, y_test, cg_name=cg_name, optimal=optimal, v=v)
            return
        else:
            fitter = self.fit_linear()
        x_to_fit = np.linspace(np.nanmin(self.x), np.nanmax(self.x), 300)
        fitted = fitter.predict(x_to_fit)
        test_mask = ~np.isnan(x_test) & ~np.isnan(y_test)
        utils.plot_graph_nicely([x, x_to_fit, x_test], [y, fitted, y_test],
                                cg_name + " " + name_of_type[type] + ", loss = " + str(
                                    round(fitter.score(x_test[test_mask], y_test[test_mask]), 2)))
