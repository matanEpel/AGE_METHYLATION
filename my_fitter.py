import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

import utils

SPLIT_SIZE = 7


# TODO - make this fitter better (for edge cases)

class MY_FITTER:
    def __init__(self, x, y):
        mask = ~np.isnan(x) & ~np.isnan(y)
        self.x = x[mask]
        self.y = y[mask]
        self.start_y = 0
        self.direction = 1
        self.locations = []
        self.acordionicity = []

    def fit(self, optimal=False, v=3):
        self.start_y = 0
        self.locations = []
        self.acordionicity = []
        mask = ~np.isnan(self.x)
        x, y = self.x[mask], self.y[mask]
        splitted_x, splitted_y = utils.split_array(x, y, SPLIT_SIZE, optimal=optimal, v=v)
        for i in range(len(splitted_x)):
            accordionicity = (splitted_y[i][-1] - splitted_y[i][0]) / (splitted_x[i][-1] - splitted_x[i][0])
            self.acordionicity.append(np.abs(accordionicity))
            self.locations.append(splitted_x[i][-1])
        new_x = utils.rescale_array(x, self.acordionicity, self.locations)
        gradient, intercept, r_value, p_value, slope_std_error = stats.linregress(new_x, y)
        self.gradient = gradient
        self.intercept = intercept
        return self

    def predict(self, x=None):
        new_x = utils.rescale_array(x, self.acordionicity, self.locations)
        return self.intercept + self.gradient * new_x

    def create_age_aacordionicity_graph(self):
        x = [0] + [0, self.locations[0]] + [x for pair in zip(self.locations[1:], self.locations[1:]) for x in pair] + [self.x[-1], self.x[-1]]
        y = [0]+[self.acordionicity[0]] + [x for pair in zip(self.acordionicity, self.acordionicity) for x in pair] + [0]
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

    def create_results_graph(self, x_test, y_test, cg_name="", optimal=False,v=3):
        y = self.y
        x = self.x

        fitter = self.fit(optimal=optimal,v=v)
        x_to_fit = np.linspace(np.nanmin(self.x), np.nanmax(self.x), 300)
        fitted = fitter.predict(x_to_fit)
        test_mask = ~np.isnan(x_test) & ~np.isnan(y_test)
        name = "my"
        if optimal:
            name += " optimal"
        else:
            name += " not optimal"
        utils.plot_graph_nicely([x, x_to_fit, x_test], [y, fitted, y_test],
                                cg_name + " " + name + " v = " + str(v)+", loss = " + str(
                                    round(fitter.score(x_test[test_mask], y_test[test_mask]), 2)))

    def get_type(self):
        classes = [[0, 20], [20, 40], [40, 60], [60, 80]]
        types = [1, 2, 3, 4]

        my_type = None

        x_s = np.array(list(i for i in range(80)))
        y_s = self.predict(x_s)

        down_or_up = (y_s[-1] - y_s[0]) / np.abs(y_s[-1] - y_s[0])

        diffs = np.abs(y_s[1:] - y_s[:-1])
        total = np.sum(diffs)

        for idx, c in enumerate(classes):
            if np.sum(diffs[c[0]: c[1]]) / total > 0.4 and total > 30:
                my_type = types[idx]

        return my_type, down_or_up

    def get_vector(self):
        final_v = []
        curr_loc = 0
        for i in range(80):
            if i >= self.locations[curr_loc]:
                curr_loc += 1
            final_v.append(self.acordionicity[curr_loc]*self.direction)

        return final_v
