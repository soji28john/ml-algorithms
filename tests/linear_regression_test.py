# test_linear_regression.py
"""
Unit tests for the custom LinearRegression class.
Run from terminal using: pytest tests/test_linear_regression.py
"""
import pytest
import numpy as np


from algorithms.linear_regression_model import LinearRegression 
from sklearn.metrics import r2_score as sk_r2_score 


@pytest.fixture(scope='module')
def simple_linear_data():
    """Fixture to provide a simple, perfect linear dataset (y = 2x)."""
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    return X, y

@pytest.fixture(scope='module')
def complex_linear_data():
    """Fixture to provide a dataset with multiple features."""
    X = np.array([[1, 5], [2, 6], [3, 7], [4, 8]])
    y = np.array([11, 14, 17, 20]) # y = 3x1 + x2 + 3 (approx)
    return X, y

# --- Unit Tests ---

def test_initialization_defaults():
    """Test that the model initializes with default parameters."""
    model = LinearRegression()
    assert model.learning_rate == 0.01
    assert model.n_iterations == 1000
    assert model.weights is None
    assert model.bias is None
    assert model.cost_history == []

def test_fit_initializes_parameters(simple_linear_data):
    """Test that fit() initializes weights and bias correctly after execution."""
    X, y = simple_linear_data
    model = LinearRegression(n_iterations=10) # Use few iterations for speed
    model.fit(X, y)
    
    assert model.weights is not None
    assert model.weights.shape == (X.shape[1],) 
    assert model.bias is not None
    assert len(model.cost_history) == 10

def test_predict_shape(simple_linear_data):
    """Test that predict() returns the correct number of predictions."""
    X, y = simple_linear_data
    model = LinearRegression(n_iterations=10)
    model.fit(X, y)
    y_pred = model.predict(X)
    
    assert y_pred.shape == y.shape
    assert isinstance(y_pred, np.ndarray)

def test_model_convergence_simple(simple_linear_data):
    """Test if the model finds the correct parameters for a perfect dataset."""
    X, y = simple_linear_data
    # For y = 2x + 0, weights should be 2, bias should be 0
    model = LinearRegression(learning_rate=0.01, n_iterations=2000)
    model.fit(X, y)
    
    # Check if the weight is close to 2 and bias is close to 0
    np.testing.assert_allclose(model.weights[0], 2.0, atol=0.01)
    np.testing.assert_allclose(model.bias, 0.0, atol=0.05)
    
    # Final cost should be very close to zero
    assert model.cost_history[-1] < 1e-4

def test_model_convergence_multi_feature(complex_linear_data):
    """Test fitting with multiple features."""
    X, y = complex_linear_data
    model = LinearRegression(learning_rate=0.005, n_iterations=5000)
    model.fit(X, y)
    
    # Just check for low final cost as parameter values are harder to determine exactly
    assert model.cost_history[-1] < 0.1, "Model failed to converge to a low cost."
    assert model.weights.shape == (2,), "Weights should match 2 features."

def test_score_calculation(simple_linear_data):
    """Test that the custom score() method calculates the R^2 correctly."""
    X, y = simple_linear_data
    model = LinearRegression(learning_rate=0.01, n_iterations=2000)
    model.fit(X, y)
    
    # Custom R^2 score
    custom_r2 = model.score(X, y)
    
    # Sklearn's R^2 score 
    y_pred = model.predict(X)
    sklearn_r2 = sk_r2_score(y, y_pred)
     
    np.testing.assert_allclose(custom_r2, sklearn_r2, atol=1e-6)
       
    assert custom_r2 > 0.9999