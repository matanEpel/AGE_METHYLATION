import utils
from few_examples_for_fitting import examples
from fitting_models import Fitter, LINEAR, LOG, MY_METHOD
from research import analysis, calculate_mse_stats, clustering


def main():
    # examples(plot=False, amount =2374)
    clustering()


if __name__ == '__main__':
    main()
