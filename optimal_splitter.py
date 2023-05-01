from copy import copy

import utils
import numpy as np
from scipy import stats
import random
AMOUNT_OF_RANDOM_LOC = 100


class OPTIMAL_SPLITTER:
    def __init__(self):
        self.locations = []

    def get_score(self, splitted_x, splitted_y,x=None, y=None):
        locations = []
        acordionicity = []
        not_have_x = False
        if x is None:
            x = []
            y = []
            not_have_x = True
        for i in range(len(splitted_x)):
            accordionicity = (splitted_y[i][-1] - splitted_y[i][0]) / (splitted_x[i][-1] - splitted_x[i][0])
            acordionicity.append(np.abs(accordionicity))
            locations.append(splitted_x[i][-1])
            if not_have_x:
                x += splitted_x[i][:-1]
                y += splitted_y[i][:-1]
        if not_have_x:
            x = np.array(x + [splitted_x[-1][-1]])
            y = np.array(y + [splitted_y[-1][-1]])
        new_x = utils.rescale_array(x, acordionicity, locations)
        gradient, intercept, r_value, p_value, slope_std_error = stats.linregress(new_x, y)

        return np.average(np.power(y - (new_x * gradient + intercept), 2))

    def split_v1(self, x, y, split_size):
        # optimize by basic split and then moving to different directions
        curr_splitted_x, curr_splitted_y = utils.split_array(x, y, SPLIT_SIZE=split_size, optimal=False)
        curr_score = self.get_score(curr_splitted_x, curr_splitted_y)

        new_score, new_split = self.get_next_step_best(curr_splitted_x, curr_splitted_y, x, y)
        while new_score < curr_score:
            curr_score = new_score
            curr_splitted_x, curr_splitted_y = new_split[0], new_split[1]
            new_score, new_split = self.get_next_step_best(curr_splitted_x, curr_splitted_y, x, y)
        return curr_splitted_x, curr_splitted_y

    def split_v3(self, x, y, split_size):
        optional_locations = self.generate_optional_locations(split_size, np.max(x) + 1)
        best_score = np.inf
        best_splitted_x, best_splitted_y = None, None
        for locations in optional_locations:
            best_inner_score = np.inf
            best_inner_splitted_x, best_inner_splitted_y = None, None
            curr_splitted_x, curr_splitted_y = self.get_split_by_locations(x, y, locations)
            curr_score = self.get_score(curr_splitted_x, curr_splitted_y, x, y)
            new_locations = locations

            while curr_score < best_inner_score:
                best_inner_splitted_x, best_inner_splitted_y = curr_splitted_x, curr_splitted_y
                best_inner_score = curr_score
                new_locations = self.get_next_best_location(x, y, new_locations)
                curr_splitted_x, curr_splitted_y = self.get_split_by_locations(x, y, new_locations)
                curr_score = self.get_score(curr_splitted_x, curr_splitted_y, x, y)

            if best_inner_score < best_score:
                best_splitted_x, best_splitted_y = best_inner_splitted_x, best_inner_splitted_y
                best_score = best_inner_score

        return best_splitted_x, best_splitted_y

    def split_v2(self, x, y, split_size):
        # optimize by trying 100 combinations of locations
        optional_locations = self.generate_optional_locations(split_size, np.max(x)+1)
        best_score = np.inf
        best_splitted_x, best_splitted_y = None, None
        for locations in optional_locations:
            curr_splitted_x, curr_splitted_y = self.get_split_by_locations(x, y, locations)
            curr_score = self.get_score(curr_splitted_x, curr_splitted_y, x, y)
            if curr_score < best_score:
                best_splitted_x, best_splitted_y = curr_splitted_x, curr_splitted_y
        return best_splitted_x, best_splitted_y

    def get_split_by_locations(self, x, y, locations):
        locations += [np.inf]
        splitted_x, splitted_y = [[]], [[]]
        location_idx = 0
        for idx, x_val in enumerate(x):
            if x_val < locations[location_idx]:
                splitted_x[-1].append(x[idx])
                splitted_y[-1].append(y[idx])
            else:
                splitted_x[-1].append(x[idx])
                splitted_y[-1].append(y[idx])
                location_idx += 1
                splitted_x.append([x[idx]])
                splitted_y.append([y[idx]])
        locations.remove(np.inf)
        return splitted_x, splitted_y

    def get_next_step_best(self, curr_splitted_x, curr_splitted_y, x, y):
        score = np.inf
        split = (curr_splitted_x, curr_splitted_y)
        for i in range(len(curr_splitted_x) - 1):
            opt1, opt2 = self.get_switch_in_location(curr_splitted_x, curr_splitted_y, i)
            opt3, _ = self.get_switch_in_location(opt1[0], opt1[1], i)
            _, opt4 = self.get_switch_in_location(opt2[0], opt2[1], i)

            if self.get_score(opt1[0], opt1[1], x, y) < score:
                score = self.get_score(opt1[0], opt1[1], x, y)
                split = opt1
            if self.get_score(opt2[0], opt2[1], x, y) < score:
                score = self.get_score(opt2[0], opt2[1], x, y)
                split = opt2
            if self.get_score(opt3[0], opt3[1], x, y) < score:
                score = self.get_score(opt3[0], opt3[1], x, y)
                split = opt3
            if self.get_score(opt4[0], opt4[1], x, y) < score:
                score = self.get_score(opt4[0], opt4[1], x, y)
                split = opt4
        return score, split

    def get_switch_in_location(self, curr_splitted_x, curr_splitted_y, i):
        opt1 = (copy(curr_splitted_x), copy(curr_splitted_y))
        opt2 = (copy(curr_splitted_x), copy(curr_splitted_y))

        if len(opt1[1][i + 1]) > 2 and len(opt1[0][i + 1]) > 2:
            opt1[0][i] = np.append(opt1[0][i], opt1[0][i + 1][1])
            opt1[0][i + 1] = opt1[0][i + 1][1:]
            opt1[1][i] = np.append(opt1[1][i], opt1[1][i + 1][1])
            opt1[1][i + 1] = opt1[1][i + 1][1:]

        if len(opt2[1][i]) > 2 and len(opt2[0][i]) > 2:
            opt2[0][i] = opt2[0][i][:-1]
            opt2[0][i + 1] = np.concatenate([np.array([opt2[0][i][-1]]), opt2[0][i + 1]]).tolist()
            opt2[1][i] = opt2[1][i][:-1]
            opt2[1][i + 1] = np.concatenate([np.array([opt2[1][i][-1]]), opt2[1][i + 1]]).tolist()

        return opt1, opt2

    def generate_optional_locations(self, split_size, max_x):
        opt_locations = []
        for _ in range(AMOUNT_OF_RANDOM_LOC):
            opt_locations.append(sorted(random.sample(range(0, max_x),split_size)))
        return  opt_locations

    def get_next_best_location(self, x, y, locations):
        optional_locs = []
        for i in range(len(locations)):
            opt1 = copy(locations)
            opt2 = copy(locations)
            opt1[i] += 1
            opt2[i] -= 1
            optional_locs.append(opt1)
            optional_locs.append(opt2)
        best_score = np.inf
        best_loc = locations
        for optional_loc in optional_locs:
            curr_splitted_x, curr_splitted_y = self.get_split_by_locations(x, y, optional_loc)
            curr_score = self.get_score(curr_splitted_x, curr_splitted_y, x, y)
            if curr_score < best_score:
                best_loc = optional_loc
                best_score = curr_score
        return best_loc
