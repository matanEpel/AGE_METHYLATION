import numpy as np
from matplotlib import pyplot as plt

import utils
from clustering import CLUSTER, AMOUNT_OF_GROUPS
from fitting_models import Fitter, MY_METHOD, LINEAR, LOG, MAGE

types = {
    1: [77, 78, 120, 123, 143, 166, 177, 207, 227, 274, 295, 296, 342, 353, 385, 390, 399, 421, 434, 440, 481, 503, 509,
        526, 538, 557, 607, 691, 704, 710, 747, 769, 819, 857, 865, 869, 877, 896, 945, 987, 1040, 1069, 1075, 1101,
        1102, 1124, 1126, 1135, 1148, 1168, 1175, 1183, 1187, 1189, 1193, 1202, 1225, 1273, 1335, 1347, 1414, 1475,
        1477, 1510, 1512, 1514, 1582, 1589, 1632, 1634, 1649, 1679, 1695, 1709, 1722, 1724, 1735, 1742, 1759, 1764,
        1802, 1825, 1866, 1883, 1894, 1922, 1926, 1933, 1965, 1970, 1996, 1998, 2002, 2025, 2032, 2045, 2094, 2135,
        2141, 2154, 2158, 2169, 2190, 2196, 2197, 2207, 2213, 2307, 2357, 2367], 2: [525, 803, 1076], 3: [], 4: []}


def analysis():
    ages, train, test, cg_names = utils.get_data()

    for i in [types[1][0]] + types[2]:
        y = train[i]
        y_test = test[i]
        x = ages
        fitter = Fitter(x, y)
        fitter.create_results_graph(x, y_test, cg_names[i], type=MY_METHOD, optimal=True, v=1)
        fitter.create_age_aacordionicity_graph()
        fitter.create_results_graph(x, y_test, cg_names[i], type=MY_METHOD, optimal=True, v=2)
        fitter.create_age_aacordionicity_graph()
        fitter.create_results_graph(x, y_test, cg_names[i], type=MY_METHOD, optimal=True, v=3)
        a = fitter.get_vector()
        fitter.create_age_aacordionicity_graph()
        fitter.create_results_graph(x, y_test, cg_names[i], type=LINEAR)
        fitter.create_results_graph(x, y_test, cg_names[i], type=LOG)
        fitter.create_results_graph(x, y_test, cg_names[i], type=MAGE)


# average error over interesting points is:
# lin error: 3.719
# log error: 2.528
# mAge: 2.642
# for 10 random points: 2.02
# for 50: 1.92
# for 100: 1.91
def calculate_mse_stats():
    interesting_points, interesting_cgs = utils.get_interesting_points()
    ages, train, test, cg_names = utils.get_data()
    errors_our = []
    errors_lin = []
    errors_log = []
    for i in interesting_points:
        print(i)
        y = train[i]
        y_test = test[i]
        x = ages
        fitter = Fitter(x, y)
        # fitter.create_results_graph(x, y_test, type=MAGE)
        # fitter.fit_mAge()
        # score = fitter.score(x, y_test)
        # if score == score:
        #     errors_our.append(score)
        #     print(errors_our[-1])
        fitter.fit_my(optimal=True, v=3)
        score = fitter.score(x, y_test)
        if score == score:
            errors_our.append(score)
            print(errors_our[-1])
        # fitter.fit_linear()
        # score = fitter.score(x, y_test)
        # if score != np.nan:
        #     errors_lin.append(score)
        # fitter.fit_log()
        # score = fitter.score(x, y_test)
        # if score != np.nan:
        #     errors_log.append(score)
        #     print(errors_our[-1])
    print(np.sum(errors_our) / len(errors_our))
    # print(np.sum(errors_lin)/len(errors_lin))
    # print(np.sum(errors_log)/len(errors_log))


def clustering():
    interesting_points, interesting_cgs = utils.get_interesting_points()
    ages, train, test, cg_names = utils.get_data()
    vectors = []
    for i in range(1000):
        print(i)
        y = train[i]
        x = ages
        fitter = Fitter(x, y)
        fitter.fit_my(optimal=True, v=3)
        vectors.append(fitter.get_vector())
    vectors = np.array(vectors)
    np.save("vectors.npy", vectors)
    # vectors = np.load("vectors.npy")
    vectors = vectors[~np.isnan(vectors).any(axis=1)]
    # 165, 174, 206
    c = CLUSTER()
    groups, is_clustered, vectors = c.cluster_pca(vectors)
    groups = groups[is_clustered]
    plt.hist(groups, bins=AMOUNT_OF_GROUPS)
    plt.show()
    plt.plot(vectors[0])
    plt.show()
    plt.plot(vectors[-3])
    plt.show()