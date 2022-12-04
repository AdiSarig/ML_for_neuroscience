import numpy as np
from sklearn.linear_model import ElasticNet

from final.grid_model_search import grid_model_search, plot_model_search

L1_RATIOS = np.linspace(0, 1, 100)  # l1_ration = 0 -> using just ridge, l1_ration = 1 -> using just Lasso
ALPHAS = np.linspace(1, 50, 100)
PARAM_GRID = {"alpha": ALPHAS, "l1_ratio": L1_RATIOS}
ESTIMATOR = ElasticNet(random_state=0, max_iter=10000, tol=0.5)

MODEL_SEARCH_FIG_PATH = './model_search_linear.jpg'


def get_best_linear_model(x_train, y_train):
    """
    Tune hyperparameters such as penalty and mixing parameter ratio, and look for the best elastic net model using CV

    Args:
        x_train: predictors - molecular decomposition
        y_train: odor perception

    Returns:
        model_searcher: the best model using the best parameters
    """
    print('Search for best linear model')
    model_searcher = grid_model_search(x_train, y_train, PARAM_GRID, ESTIMATOR)
    x_header = 'param_alpha'
    y_header = 'param_l1_ratio'
    plot_model_search(model_searcher, x_header, y_header, MODEL_SEARCH_FIG_PATH)
    return model_searcher
