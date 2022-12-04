import shap
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from final.decission_trees import get_best_non_linear_model
from final.globals import ODOR_DILUTION
from final.linear_regression import get_best_linear_model


def build_models(df, odor, features):
    """
    Build linear and non-linear models, test them against hyper-parameters and chose the best model based on the
    MSE results from the test data.
    Args:
        df: df with all the data
        odor: odor name ad the target variable
        features: list of features name

    Returns:
        - best model based on hte test data
        - mse of the model based on test data
        - x_train data
    """
    x_train, x_test, y_train, y_test = split_and_standardize(df, odor, features)

    # build regression model - ElasticNet
    linear_model = get_best_linear_model(x_train, y_train)

    # build decision tree model - gradient boosting
    decision_tree_model = get_best_non_linear_model(x_train, y_train)

    print('Test models using test dataset')
    mse_linear = test_model(linear_model, x_test, y_test)
    mse_non_linear = test_model(decision_tree_model, x_test, y_test)

    print(f'MSE for different models:')
    print(f'Linear: {mse_linear}')
    print(f'Non-linear: {mse_non_linear}')

    if mse_linear > mse_non_linear:
        print(f'Chosen model is: non linear model. Test MSE: {mse_non_linear}')
        return decision_tree_model, mse_non_linear, x_train
    else:
        print(f'Chosen model is: linear model. Test MSE: {mse_linear}')
        return linear_model, mse_linear, x_train


def split_and_standardize(df, odor, feature_names):
    """
    Prepare the data for the model with splits data, standardize numeric features, dummy coding categoriel features
    and drop rows with Na.
    Args:
        df: df to work with
        odor: odor name for the model target
        feature_names: list of features for the model

    Returns: x_train, x_test, y_train, y_test

    """
    # remove rows where the target is nan
    odor_df = df.dropna(subset=[odor])
    # remove rows where the features are nan
    odor_df = odor_df.dropna(subset=feature_names)

    # split to train and test
    x = odor_df[feature_names]
    y = odor_df[odor].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

    # standardize only numeric features (exclude dilution for example)
    numeric_features = x_train.select_dtypes([int, float])

    scaler = StandardScaler()
    # standardize train data
    x_train.loc[:, numeric_features.columns] = scaler.fit_transform(x_train.loc[:, numeric_features.columns])
    # standardize test data
    x_test.loc[:, numeric_features.columns] = scaler.fit_transform(x_test.loc[:, numeric_features.columns])

    # set dummy coding
    x_train = x_train.astype({ODOR_DILUTION: int})
    x_test = x_test.astype({ODOR_DILUTION: int})

    return x_train, x_test, y_train, y_test


def test_model(model, x_test, y_test):
    """
    Calculate MSE for the given model
    Args:
        model: input model
        x_test: list of features to test
        y_test: test target values

    Returns:
        MSE for the y_test and y_pred of the model
    """
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


def explainability(model, x):
    """
    Plot SHAP values and force plot for the chosen model.
    Args:
        model: model object
        x: features data
    """
    shap.initjs()
    explainer = shap.KernelExplainer(model=model.predict, data=x, link="identity")
    shap_values = explainer.shap_values(X=x, nsamples=100)
    shap.summary_plot(shap_values=shap_values, features=x, max_display=20, plot_size=[20, 10])
