import os
import pytest
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

if os.getenv("_PYTEST_RAISE", "0") != "0":
    # For debugging tests with pytest and vscode:
    # Configure pytest to not swallow exceptions, so that vscode can catch them before the debugging session ends.
    # See https://stackoverflow.com/a/62563106/130164
    # The .vscode/launch.json configuration should be:
    # "configurations": [
    #     {
    #         "name": "Python: Debug Tests",
    #         "type": "python",
    #         "request": "launch",
    #         "program": "${file}",
    #         "purpose": ["debug-test"],
    #         "console": "integratedTerminal",
    #         "justMyCode": false,
    #         "env": {
    #             "_PYTEST_RAISE": "1"
    #         },
    #     },
    # ]
    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value


# Fixtures


def models_factory():
    return {
        "dummy": DummyClassifier(strategy="stratified"),
        "logistic_multinomial": LogisticRegression(multi_class="multinomial"),
        "logistic_ovr": LogisticRegression(multi_class="ovr"),
        "randomforest": RandomForestClassifier(),
        "linearsvm": SVC(kernel="linear"),
        "nonlinear_svm": SVC(kernel="rbf"),
    }


all_model_names = list(models_factory().keys())
tree_models = ["randomforest"]
# multiclass OvR:
ovr_models = ["logistic_multinomial", "logistic_ovr"]
# multiclass OvO:
ovo_models = ["linearsvm"]
# no coefs_, feature_importances_, or feature names:
dummy_models = ["dummy"]
# no coefs_ or feature_importances_ but does have feature names:
no_coefs_models = ["nonlinear_svm"]


@pytest.fixture(name="models_factory")
def models_factory_fixture():
    # Pass the lambda function factory as a fixture
    # https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly
    return models_factory


@pytest.fixture
def sample_data():
    """Make multiclass train and test data"""
    n_features = 5
    covid_data = np.random.randn(100, n_features) + np.array([5] * n_features)
    hiv_data = np.random.randn(100, n_features) + np.array([15] * n_features)
    healthy_data = np.random.randn(100, n_features) + np.array([-5] * n_features)
    # add a fourth class, so that coefs_ for OvO and OvR models have different shapes.
    ebola_data = np.random.randn(100, n_features) + np.array([-15] * n_features)
    X_train = np.vstack([covid_data, hiv_data, healthy_data, ebola_data])
    y_train = np.hstack(
        [
            ["Covid"] * covid_data.shape[0],
            ["HIV"] * hiv_data.shape[0],
            ["Healthy"] * healthy_data.shape[0],
            ["Ebola"] * ebola_data.shape[0],
        ]
    )
    X_test = np.array(
        [
            [5] * n_features,
            [15] * n_features,
            [-5] * n_features,
            [6] * n_features,
            [-15] * n_features,
        ]
    )
    y_test = np.array(["Covid", "HIV", "Healthy", "Covid", "Ebola"])
    return (X_train, y_train, X_test, y_test)


@pytest.fixture
def sample_data_two(sample_data):
    """Same train data, different test data"""
    (X_train, y_train, X_test, y_test) = sample_data
    n_features = X_train.shape[1]
    X_test2 = np.array(
        [
            [6] * n_features,
            [14] * n_features,
            [-4] * n_features,
            [4] * n_features,
            [-14] * n_features,
        ]
    )
    y_test2 = np.array(["Covid", "HIV", "Healthy", "Covid", "Ebola"])
    return (X_train, y_train, X_test2, y_test2)
