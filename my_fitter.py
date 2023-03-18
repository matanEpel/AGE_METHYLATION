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

    def fit(self,optimal=False):
        self.start_y = 0
        self.locations = []
        self.acordionicity = []
        mask = ~np.isnan(self.x)
        x, y = self.x[mask], self.y[mask]
        splitted_x, splitted_y = utils.split_array(x,y, SPLIT_SIZE, optimal=optimal)
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
        x = [0, self.locations[0]] + [x for pair in zip(self.locations[1:],self.locations[1:]) for x in pair]+[100]
        y = [self.acordionicity[0]]+[x for pair in zip(self.acordionicity,self.acordionicity) for x in pair]
        plt.plot(x, y)
        plt.title("acordionicity by age")
        plt.xlabel("age")
        plt.ylabel("acordionicity")
        plt.legend()
        plt.show()
