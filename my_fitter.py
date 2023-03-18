import numpy as np
from matplotlib import pyplot as plt

import utils

SPLIT_SIZE = 7


class MY_FITTER:
    def __init__(self, x, y):
        mask = ~np.isnan(x) & ~np.isnan(y)
        self.x = x[mask]
        self.y = y[mask]
        self.start_y = 0
        self.direction = 1
        self.locations = []
        self.acordionicity = []

    def fit(self, optimal=False):
        self.start_y = 0
        self.locations = []
        self.acordionicity = []
        mask = ~np.isnan(self.x)
        x, y = self.x[mask], self.y[mask]
        splitted_x, splitted_y = utils.split_array(x, y, SPLIT_SIZE, optimal=optimal)
        for i in range(len(splitted_x)):
            accordionicity = (splitted_y[i][-1] - splitted_y[i][0]) / (splitted_x[i][-1] - splitted_x[i][0])
            self.acordionicity.append(np.abs(accordionicity))
            self.locations.append(splitted_x[i][-1])
            self.start_y = self.y[mask][0]
        if y[-1] > y[0]:
            self.direction = 1
        else:
            self.direction = -1
        return self

    def predict(self, x=None):
        new_x = utils.rescale_array(x, self.acordionicity, self.locations)
        return self.start_y + self.direction * new_x

    def create_age_aacordionicity_graph(self):
        x = [0, self.locations[0]] + [x for pair in zip(self.locations[1:], self.locations[1:]) for x in pair] + [100]
        y = [self.acordionicity[0]] + [x for pair in zip(self.acordionicity, self.acordionicity) for x in pair]
        plt.plot(x, y)
        plt.title("acordionicity by age")
        plt.xlabel("age")
        plt.ylabel("acordionicity")
        plt.show()

    def score(self, x, y):
        """returns the mse"""
        mask = ~np.isnan(x) & ~np.isnan(y)
        mask = mask & ~np.isnan(self.predict(x))
        predicted_y = self.predict(x[mask])
        return np.sqrt(np.average(np.power(y[mask] - predicted_y, 2)))

    def create_results_graph(self, x_test, y_test, cg_name="", optimal=False):
        y = self.y
        x = self.x

        fitter = self.fit()
        x_to_fit = np.linspace(np.nanmin(self.x), np.nanmax(self.x), 300)
        fitted = fitter.predict(x_to_fit)
        test_mask = ~np.isnan(x_test) & ~np.isnan(y_test)
        name = "my"
        if optimal:
            name += " optimal"
        else:
            name += " not optimal"
        utils.plot_graph_nicely([x, x_to_fit, x_test], [y, fitted, y_test],
                                cg_name + " " + name + ", loss = " + str(
                                    round(fitter.score(x_test[test_mask], y_test[test_mask]), 2)))

    def get_type(self):
        my_type = None

        return my_type