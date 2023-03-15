import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io


def get_12K_data():
    return scipy.io.loadmat('data/12K.mat')


def get_interesting_points():
    interesting_points = []
    interesting_cgs = []
    ages, train, test, cg_names = get_data()
    for i in range(train.shape[0]):
        if np.nanmax(train[i]) - np.nanmin(train[i]) >= 55:
            interesting_points.append(i)
            interesting_cgs.append(cg_names[i])
    return interesting_points, interesting_cgs


def plot_graph_nicely(x_s, y_s, title, org=True):
    plt.scatter(x_s[0], y_s[0], c='g', marker= "o", s=30, label="data")
    if len(x_s) > 1:
        plt.plot(x_s[1], y_s[1], c='black', label="fit")
    if len(x_s) > 2:
        plt.scatter(x_s[2], y_s[2], c='r', marker= "x", s=40, linewidth = 1,label="test data")
    plt.xlim((0, 100))
    plt.ylim((0, 100))
    if org:
        plt.title(title + " original scale")

    else:
        plt.title(title + " fitted scale")
    plt.xlabel("age")
    plt.ylabel("methylation percentage")
    plt.legend()
    plt.show()


def get_data():
    data = get_12K_data()
    ages = data["XX"]
    train = data["MM"]
    test = data["TT"]
    cg_name = data["cgs"]
    cg_name = np.array([a[0][0] for a in cg_name])
    return ages.flatten(), train, test, cg_name.flatten()
