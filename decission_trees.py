from sklearn.ensemble import GradientBoostingRegressor

from final.grid_model_search import grid_model_search, plot_model_search

N_ESTIMATORS = range(1, 60, 2)
MAX_DEPTH = range(1, 10)
PARAM_GRID = {"n_estimators": N_ESTIMATORS, "max_depth": MAX_DEPTH}
ESTIMATOR = GradientBoostingRegressor(random_state=0)

MODEL_SEARCH_FIG_PATH = './model_search_non_linear.jpg'


def get_best_non_linear_model(x_train, y_train):
    """
    Tune hyperparameters such as max depth and look for the best gradient boosting model using CV

    Args:
        x_train: predictors - molecular decomposition
        y_train: odor perception

    Returns:
        model_searcher: the best model using the best parameters
    """
    print('Search for best non-linear model')
    model_searcher = grid_model_search(x_train, y_train, PARAM_GRID, ESTIMATOR)
    x_header = 'param_n_estimators'
    y_header = 'param_max_depth'
    plot_model_search(model_searcher, x_header, y_header, MODEL_SEARCH_FIG_PATH)
    return model_searcher
