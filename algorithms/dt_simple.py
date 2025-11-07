"""
Decision Tree on Simple Synthetic Data 2 Features
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

def load_synthetic_data(n_samples=50):
    """Loads simple synthetic data for binary classification."""
    X, y = make_classification(n_samples=n_samples, n_features=2, 
                               n_informative=2, n_redundant=0, 
                               random_state=42)
    return X, y

def train_synthetic_dt_model(X, y, test_size=0.3, random_state=42):
    """Splits data and trains a Decision Tree model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_synthetic_model(model, X_test, y_test):
    """Evaluates the model and returns accuracy and classification report."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return accuracy, report
if __name__ == '__main__':
    # Load the data
    X, y = load_synthetic_data()
    
    # Train the model and get test data
    model, X_test, y_test = train_synthetic_dt_model(X, y, test_size=0.3)
    
    # Evaluate the model
    accuracy, report = evaluate_synthetic_model(model, X_test, y_test)
    
    # Print the result to the terminal
    print("\n--- Synthetic Data Decision Tree Execution ---")
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print("Classification Report Summary:")
    