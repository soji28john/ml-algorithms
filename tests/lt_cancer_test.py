# tests/test_cancer_model.py
"""
Unit tests for the Logistic Regression model on the Breast Cancer dataset.
Run from terminal using: pytest tests/test_cancer_model.py
"""
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression

from algorithms.logistic_regression_cancer import load_cancer_data, train_and_evaluate_cancer_model

# Setup fixture to load data once for all tests
@pytest.fixture(scope='module')
def cancer_data():
    """Fixture to load Breast Cancer data once per module."""
    return load_cancer_data()

def test_cancer_data_loading_shape(cancer_data):
    """Test that the loaded data has the correct number of samples (569) and features (30)."""
    X, y, _ = cancer_data
    assert X.shape == (569, 30)
    assert y.shape == (569,)

def test_cancer_model_is_fitted(cancer_data):
    """Test that the training function returns a properly fitted model."""
    X, y, _ = cancer_data
    model, _, _, _ = train_and_evaluate_cancer_model(X, y)
    
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, 'coef_')
    assert model.coef_.shape == (1, 30), "Coefficients shape should match 30 features."

def test_cancer_model_minimum_accuracy_sanity_check(cancer_data):
    """Test for a high minimum accuracy (e.g., 90%) on the test set."""
    X, y, _ = cancer_data
    _, accuracy, _, _ = train_and_evaluate_cancer_model(X, y, random_state=1)
    
    # We expect high performance (> 0.90) for a well-tuned LR model on this dataset
    assert accuracy > 0.90, f"Accuracy of {accuracy:.4f} is unexpectedly low."

def test_cancer_model_low_false_negatives(cancer_data):
    """Test that the critical False Negative count is acceptably low (e.g., < 10)."""
    X, y, _ = cancer_data
    _, _, _, conf_mat = train_and_evaluate_cancer_model(X, y, random_state=1)
    
    # False Negatives (actual Malignant, predicted Benign) is the bottom-left element
    false_negatives = conf_mat[1, 0] 
    
    # Check that false negatives are not excessive (e.g., < 10 for a 30% split)
    assert false_negatives < 10, f"False Negative count ({false_negatives}) is too high."