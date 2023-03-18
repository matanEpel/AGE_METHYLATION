import numpy as np

import utils

SPLIT_SIZE = 8


class MY_FITTER:
    def __init__(self, x, y):
        mask = ~np.isnan(x) & ~np.isnan(y)
        self.x = x[mask]
        self.y = y[mask]
        self.start_y = 0
        self.direction = 1
        self.locations = []
        self.acordionicity = []

    def fit(self):
        mask = ~np.isnan(self.x)
        x, y = self.x[mask], self.y[mask]
        splitted_x, splitted_y = utils.split_array(x, SPLIT_SIZE), utils.split_array(y, SPLIT_SIZE)
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

    def predict(self, x):
        new_x = utils.rescale_array(x, self.acordionicity, self.locations)
        return self.start_y + self.direction * new_x
