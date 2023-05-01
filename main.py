import utils
from few_examples_for_fitting import examples
from fitting_models import Fitter, LINEAR, LOG, MY_METHOD


def main():
    # examples(plot=False, amount =2374)
    ages, train, test, cg_names = utils.get_data()
    y = train[1076]
    y_test = test[1076]
    x = ages
    fitter = Fitter(x, y)
    fitter.create_results_graph(x, y_test, cg_names[1076], type=MY_METHOD, optimal=True,v=1)
    fitter.create_results_graph(x, y_test, cg_names[1076], type=MY_METHOD, optimal=True,v=2)
    fitter.create_results_graph(x, y_test, cg_names[1076], type=MY_METHOD, optimal=True,v=3)


if __name__ == '__main__':
    main()
