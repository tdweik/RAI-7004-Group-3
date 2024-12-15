import sys
sys.path.append('src')
import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, r2_score
from src.algorithms.sklearn_algorithms import SklearnAlgorithm

@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return X, y

@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    return X, y

def test_logistic_regression_fit_predict(classification_data):
    X, y = classification_data
    algo = SklearnAlgorithm('logistic_regression')
    algo.fit(X, y)
    predictions = algo.predict(X)
    assert accuracy_score(y, predictions) > 0.4

def test_decision_tree_classifier_fit_predict(classification_data):
    X, y = classification_data
    algo = SklearnAlgorithm('decision_tree_classifier')
    algo.fit(X, y)
    predictions = algo.predict(X)
    assert accuracy_score(y, predictions) > 0.4

def test_random_forest_classifier_fit_predict(classification_data):
    X, y = classification_data
    algo = SklearnAlgorithm('random_forest_classifier')
    algo.fit(X, y)
    predictions = algo.predict(X)
    assert accuracy_score(y, predictions) > 0.4

def test_linear_regression_fit_predict(regression_data):
    X, y = regression_data
    algo = SklearnAlgorithm('linear_regression')
    algo.fit(X, y)
    predictions = algo.predict(X)
    assert r2_score(y, predictions) > 0.4

def test_decision_tree_regressor_fit_predict(regression_data):
    X, y = regression_data
    algo = SklearnAlgorithm('decision_tree_regressor')
    algo.fit(X, y)
    predictions = algo.predict(X)
    assert r2_score(y, predictions) > 0.4

def test_random_forest_regressor_fit_predict(regression_data):
    X, y = regression_data
    algo = SklearnAlgorithm('random_forest_regressor')
    algo.fit(X, y)
    predictions = algo.predict(X)
    assert r2_score(y, predictions) > 0.4

def test_unsupported_algorithm():
    with pytest.raises(ValueError, match="Unsupported algorithm: fake_classifier"):
        SklearnAlgorithm('fake_classifier')

def test_save_model(classification_data, tmp_path):
    X, y = classification_data
    algo = SklearnAlgorithm('logistic_regression')
    algo.fit(X, y)
    file_path = tmp_path / "model.joblib"
    algo.save_model(file_path)
    assert file_path.exists()

def test_score(classification_data):
    X, y = classification_data
    algo = SklearnAlgorithm('logistic_regression')
    algo.fit(X, y)
    score = algo.score(X, y)
    assert score > 0.7