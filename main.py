import utils
from fitting_models import Fitter, LINEAR, LOG, MY_METHOD


def main():
    ages, train, test, cg_names = utils.get_data()
    points, cg_good_names = utils.get_interesting_points()

    for i in range(5):
        y = train[points[i]]
        y_test = test[points[i]]
        x = ages
        fitter = Fitter(x, y)
        fitter.create_results_graph(x, y_test, cg_good_names[i], type=MY_METHOD, optimal=True)
        fitter.create_results_graph(x, y_test, cg_good_names[i], type=MY_METHOD, optimal=False)


if __name__ == '__main__':
    main()
