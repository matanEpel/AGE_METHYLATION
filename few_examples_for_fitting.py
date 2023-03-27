import utils
from fitting_models import Fitter, LINEAR, LOG, MY_METHOD


def examples():
    ages, train, test, cg_names = utils.get_data()
    points, cg_good_names = utils.get_interesting_points()
    types = {1: [], 2: [], 3: [], 4: []}

    for i in range(3):
        # y = train[33]
        # y_test = test[33]
        y = train[points[i]]
        y_test = test[points[i]]

        x = ages
        fitter = Fitter(x, y)
        fitter.create_results_graph(x, y_test, cg_good_names[i], type=MY_METHOD, optimal=True)
        fitter.fit_my(optimal=True)
        type, direction = fitter.get_type()
        if type is not None:
            types[type].append(i)
            print("idx:", i, ", type:", type)
        # fitter.create_age_aacordionicity_graph()
        # fitter.create_results_graph(x, y_test, cg_good_names[i], type=MY_METHOD, optimal=False)
        # fitter.create_results_graph(x, y_test, cg_good_names[i], type=LINEAR, optimal=False)
        # fitter.create_results_graph(x, y_test, cg_good_names[i], type=LOG, optimal=False)
