from copy import copy

import utils
import numpy as np


class OPTIMAL_SPLITTER:
    def __init__(self):
        self.locations = []

    def get_score(self, splitted_x, splitted_y):
        score = 0
        direction = np.abs(splitted_y[-1][-1]-splitted_y[0][0])/(splitted_y[-1][-1]-splitted_y[0][0])
        for idx in range(len(splitted_y)):
            angle = np.abs((splitted_y[idx][-1] - splitted_y[idx][0]) / (splitted_x[idx][-1] - splitted_x[idx][0]))
            start_y = splitted_y[idx][0]
            start_x = splitted_x[idx][0]

            for point_idx in range(len(splitted_y[idx])):
                score += np.abs(splitted_y[idx][point_idx] - (start_y + direction*angle*(splitted_x[idx][point_idx] - start_x))) ** 2
        return score

    def split(self, x, y, split_size):
        curr_splitted_x, curr_splitted_y = utils.split_array(x, y, SPLIT_SIZE=split_size, optimal=False)
        curr_score = self.get_score(curr_splitted_x, curr_splitted_y)

        new_score, new_split = self.get_next_step_best(curr_splitted_x, curr_splitted_y)
        while new_score < curr_score:
            curr_score = new_score
            curr_splitted_x, curr_splitted_y = new_split[0], new_split[1]
            new_score, new_split = self.get_next_step_best(curr_splitted_x, curr_splitted_y)

        return curr_splitted_x, curr_splitted_y

    def get_next_step_best(self, curr_splitted_x, curr_splitted_y):
        score = np.inf
        split = (curr_splitted_x, curr_splitted_y)
        idx = 0
        for i in range(len(curr_splitted_x)-1):
            opt1, opt2 = self.get_switch_in_location(curr_splitted_x, curr_splitted_y, i)
            if self.get_score(opt1[0], opt1[1]) < score:
                idx = i
                score = self.get_score(opt1[0], opt1[1])
                split = opt1
            if self.get_score(opt2[0], opt2[1]) < score:
                score = self.get_score(opt2[0], opt2[1])
                split = opt2
                idx = i
        # print(idx)
        return score, split

    def get_switch_in_location(self, curr_splitted_x, curr_splitted_y, i):
        opt1 = (copy(curr_splitted_x), copy(curr_splitted_y))
        opt2 = (copy(curr_splitted_x), copy(curr_splitted_y))

        opt1[0][i] = np.append(opt1[0][i], opt1[0][i+1][1])
        opt1[0][i+1] = opt1[0][i+1][1:]
        opt1[1][i] = np.append(opt1[1][i],opt1[1][i+1][1])
        opt1[1][i + 1] = opt1[1][i + 1][1:]

        opt2[0][i] = opt2[0][i][:-1]
        opt2[0][i + 1] = np.concatenate([np.array([opt2[0][i][-1]]), opt2[0][i + 1]])
        opt2[1][i] = opt2[1][i][:-1]
        opt2[1][i + 1] = np.concatenate([np.array([opt2[1][i][-1]]), opt2[1][i + 1]])

        return opt1, opt2
