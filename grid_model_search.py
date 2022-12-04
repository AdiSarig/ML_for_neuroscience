import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV

SCORER = "neg_mean_squared_error"
SCORING = "neg_mean_squared_error"


def grid_model_search(x_train, y_train, param_grid, estimator):
    """
    Tune hyperparameters and look for the best model using CV based on input estimator

    Args:
        x_train: predictors - molecular decomposition
        y_train: odor perception
        param_grid: parameters for hypertuning
        estimator: which kind of model to search

    Returns:
        model_searcher: the best model using the best parameters
    """
    model_searcher = GridSearchCV(estimator, param_grid=param_grid, scoring=SCORING, refit=SCORER)
    _ = model_searcher.fit(x_train, y_train)
    print(f'The best parameters chosen: {model_searcher.best_params_}')
    print(f'The score of the best model: {model_searcher.best_score_}')

    return model_searcher


def plot_model_search(model, x_header, y_header, fig_path):
    """
    Plot the results from the grid model search.
    Args:
        model: the model searcher object
        x_header: header for the x-axis in the plot
        y_header: header for the y-axis in the plot
        fig_path: path to save the figure
    """
    # get CV results from the model searcher
    search_results = pd.DataFrame(model.cv_results_)

    x = search_results[x_header]
    y = search_results[y_header]
    z = search_results['mean_test_score']
    # convert negative MSE to positive MSE
    z = z.abs()

    color_map = plt.cm.get_cmap('Greens')
    reversed_color_map = color_map.reversed()

    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, c=z, cmap=reversed_color_map)
    ax.invert_xaxis()
    ax.set_xlabel(x_header)
    ax.set_ylabel(y_header)
    ax.set_zlabel('MSE', labelpad=10)
    plt.savefig(fig_path)
    plt.close()
