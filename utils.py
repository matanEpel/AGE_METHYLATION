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
    plt.scatter(x_s[0], y_s[0], c='g', marker="o", s=30, label="data")
    if len(x_s) > 1:
        plt.plot(x_s[1], y_s[1], c='black', label="fit")
    if len(x_s) > 2:
        plt.scatter(x_s[2], y_s[2], c='r', marker="x", s=40, linewidth=1, label="test data")
    plt.xlim((0, 100))
    plt.ylim((0, 100))
    if org:
        plt.title(title)
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


def split_array(x, SPLIT_SIZE, optimal=False):
    if not optimal:
        # the basic solution of splitting evenly
        final = []
        for i in range(SPLIT_SIZE):
            start = int(i * x.shape[0] // SPLIT_SIZE)
            if i != SPLIT_SIZE - 1:
                end = int((i + 1) * x.shape[0] // SPLIT_SIZE) + 1
            else:
                end = int(x.shape[0])
            final.append(x[start:end])
        return final
    else:
        # splitting optimally using an optimized splitter
        splitter = OPTIMAL_SPLITTER()
        return splitter.split(x, SPLIT_SIZE)


def rescale_array(x, accordionicity, locations):
    final_x = [x[0]]
    last_end = x[0]
    last_loc = last_end
    curr_limit_idx = 0
    for idx, val in enumerate(x[1:]):
        if val <= locations[curr_limit_idx] or np.isnan(val):
            final_x.append(last_loc + (val - last_end) * accordionicity[curr_limit_idx])
        else:
            last_end = locations[curr_limit_idx]
            curr_limit_idx += 1
            if curr_limit_idx >= len(locations):
                rest = np.zeros(x.shape[0]-len(final_x))
                rest.fill(np.nan)
                return np.concatenate([np.array(final_x), rest])
            last_loc = final_x[idx]
            final_x.append(last_loc + (val - last_end) * accordionicity[curr_limit_idx])
    return np.array(final_x)
