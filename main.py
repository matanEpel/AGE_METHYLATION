import utils
import numpy as np
import pandas as pd
from fitting_models import Fitter

def main():
    ages, train, test, cg_names = utils.get_data()
    points, cg_good_names = utils.get_interesting_points()

    for i in range(2):
        y = train[points[i]]
        y_test = test[points[i]]
        x = ages

        fitter = Fitter(x,y).fit_log()
        fitted = fitter.predict(x)
        print("log loss: ", fitter.score(x, y_test))
        utils.plot_graph_nicely([x, x, x], [y, fitted, y_test], cg_names[i] + " log, loss = " + str(fitter.score(x, y_test)))

        fitter = Fitter(x, y).fit_linear()
        fitted = fitter.predict(x)
        utils.plot_graph_nicely([x, x, x], [y, fitted, y_test], cg_names[i]+ " linear, loss = " + str(fitter.score(x, y_test)))
    # for idx, i in enumerate(points):
    #     utils.plot_graph_nicely([ages, ages, ages], [train[i], train[i], test[i]], cg_names[idx])


if __name__ == '__main__':
    # k12 = utils.get_12K_data()
    # print(k12)
    main()
