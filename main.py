import utils
import numpy as np
import pandas as pd
from fitting_models import Fitter, LINEAR, LOG


def main():
    ages, train, test, cg_names = utils.get_data()
    points, cg_good_names = utils.get_interesting_points()

    for i in range(2):
        y = train[points[i]]
        y_test = test[points[i]]
        x = ages
        fitter = Fitter(x, y)
        fitter.create_results_graph(x, y_test, cg_names[i], type = LINEAR)
        fitter.create_results_graph(x, y_test, cg_names[i], type = LOG)



if __name__ == '__main__':
    # k12 = utils.get_12K_data()
    # print(k12)
    main()
