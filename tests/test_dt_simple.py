# tests/test_decision_tree.py
"""
Unified unit tests for the Decision Tree classification models.
Run from the project root (ml-algorithms-scratch/) using: pytest tests/
"""
import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Import functions from the separate implementation files
from algorithms.dt_simple import load_synthetic_data, train_synthetic_dt_model
from algorithms.dt_iris import load_iris_data, train_iris_dt_model
from algorithms.dt_wine import load_wine_data, train_wine_dt_model

# --- Fixtures ---

@pytest.fixture(scope='module')
def synthetic_data():
    """Fixture for synthetic data."""
    return load_synthetic_data(n_samples=50)

@pytest.fixture(scope='module')
def iris_data():
    """Fixture for Iris data."""
    return load_iris_data()

@pytest.fixture(scope='module')
def wine_data():
    """Fixture for Wine data."""
    return load_wine_data()

# --- Tests for Data Loading and Structure ---

def test_synthetic_data_shape(synthetic_data):
    """Test synthetic data shape (50 samples, 2 features)."""
    X, y = synthetic_data
    assert X.shape == (50, 2)
    assert np.all(np.isin(y, [0, 1])), "Synthetic data should be binary."

def test_iris_data_structure(iris_data):
    """Test Iris data shape and number of classes."""
    X, y, names = iris_data
    assert X.shape == (150, 4)
    assert len(names) == 3

def test_wine_data_structure(wine_data):
    """Test Wine data shape and number of features."""
    X, y, names = wine_data
    assert X.shape == (178, 13)
    assert len(names) == 3

# --- Tests for Model Training and Sanity Checks ---

@pytest.mark.parametrize("loader, trainer", [
    (load_synthetic_data, train_synthetic_dt_model),
    (load_iris_data, train_iris_dt_model),
    (load_wine_data, train_wine_dt_model),
])
def test_all_models_are_fitted(loader, trainer):
    """Tests that every training function returns a properly fitted DT model."""
    X, y, *_ = loader() # Load data
    model, _, _ = trainer(X, y, test_size=0.1)
    
    # A fitted scikit-learn tree model must have the 'tree_' attribute
    assert isinstance(model, DecisionTreeClassifier)
    assert hasattr(model, 'tree_')
    assert model.tree_.node_count > 0, "Model did not train or is trivial."

def test_iris_performance_sanity(iris_data):
    """Test that Iris DT model achieves expected high accuracy."""
    X, y, names = iris_data
    model, X_test, y_test = train_iris_dt_model(X, y, test_size=0.2, random_state=1)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    assert accuracy >= 0.95, f"Iris DT accuracy is too low: {accuracy:.4f}"

def test_wine_prediction_shape(wine_data):
    """Test that prediction returns the correct number of outputs."""
    X, y, _ = wine_data
    model, X_test, y_test = train_wine_dt_model(X, y, test_size=0.3, random_state=1)
    y_pred = model.predict(X_test)
    
    assert y_pred.shape == y_test.shape