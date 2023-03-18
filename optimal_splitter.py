import utils


class OPTIMAL_SPLITTER:
    def __init__(self):
        self.locations = []

    def split(self, x, split_size):
        return utils.split_array(x, SPLIT_SIZE=split_size, optimal=False)