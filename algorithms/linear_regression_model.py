"""
Linear Regression from Scratch
"""
import numpy as np

class LinearRegression:
    """
    Linear Regression using Gradient Descent
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Step size for gradient descent
    n_iterations : int, default=1000
        Number of iterations for training
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X, y):
        """
        Fit linear regression model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass (predictions)
            y_predicted = self.predict(X)
            
            # Compute cost
            cost = (1 / (2 * n_samples)) * np.sum((y_predicted - y)**2)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}: Cost = {cost:.4f}")
    
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        predictions : array, shape (n_samples,)
        """
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X, y):
        """
        Calculate R² score
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
            
        Returns:
        --------
        score : float
            R² score
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum((y - y_pred)**2)
        r2_score = 1 - (ss_residual / ss_total)
        return r2_score