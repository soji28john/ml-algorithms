# unit tests
"""
Unit tests for the Logistic Regression model on the Iris binary dataset.
Run from terminal using: pytest tests/test_iris_model.py
"""
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression

# the implementation file is in the same folder or path for simple testing
from algorithms.logistic_regression_iris import load_iris_binary_data, train_and_evaluate_iris_model

# Setup fixture to load data once for all tests
@pytest.fixture(scope='module')
def iris_data():
    """Fixture to load Iris data once per module."""
    return load_iris_binary_data()

def test_iris_data_loading_shape(iris_data):
    """Test that the loaded data has the correct number of samples (100) and features (4)."""
    X, y = iris_data
    # Iris has 50 samples for class 0 and 50 for class 1 -> Total 100 samples
    assert X.shape == (100, 4)
    assert y.shape == (100,)
    assert np.all(np.isin(y, [0, 1])), "Target labels should only contain 0 and 1."

def test_iris_model_is_fitted(iris_data):
    """Test that the training function returns a properly fitted model."""
    X, y = iris_data
    model, _, _, _ = train_and_evaluate_iris_model(X, y)
    
    # A fitted scikit-learn model must have the 'coef_' and 'intercept_' attributes
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, 'coef_')
    assert hasattr(model, 'intercept_')
    assert model.coef_.shape == (1, 4), "Coefficients shape should match 4 features."

def test_iris_model_perfect_accuracy_sanity_check(iris_data):
    """Test for the expected high accuracy due to the data's linear separability."""
    X, y = iris_data
    # Use a minimal acceptable threshold (e.g., 0.98) since it should be near 1.0
    _, accuracy, _, _ = train_and_evaluate_iris_model(X, y, random_state=1)
    
    assert accuracy >= 0.98, f"Accuracy of {accuracy:.4f} is too low for this task."