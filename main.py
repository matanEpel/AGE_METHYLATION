import utils
import numpy as np
import pandas as pd


def main():
    ages, train, test, cg_names = utils.get_data()
    points, cg_names = utils.get_interesting_points()
    for idx, i in enumerate(points):
        utils.plot_graph_nicely([ages, ages, ages], [train[i], train[i], test[i]], cg_names[idx])


if __name__ == '__main__':
    # k12 = utils.get_12K_data()
    # print(k12)
    main()
