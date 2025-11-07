# logistic_regression_iris.py
"""
Logistic Regression Model Implementation for Binary Iris Classification
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_iris_binary_data():
    """
    Loads the Iris dataset and subsets it for a binary classification task.
    Returns: X (features), y (labels)
    """
    data = load_iris()
    # Use only the first two classes (0 and 1)
    X = data.data[data.target != 2]
    y = data.target[data.target != 2]
    return X, y

def train_and_evaluate_iris_model(X, y, test_size=0.3, random_state=42):
    """
    Splits data, trains Logistic Regression model, and evaluates performance.
    
    Returns: model, accuracy, report, conf_matrix
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Initialize and train the Logistic Regression model
    model = LogisticRegression(solver='liblinear', random_state=random_state)
    model.fit(X_train, y_train)

    # Prediction and Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, target_names=['setosa', 'versicolor'])
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, report, conf_matrix

if __name__ == '__main__':
    # Example execution when run directly
    X_iris, y_iris = load_iris_binary_data()
    _, acc, report, conf_mat = train_and_evaluate_iris_model(X_iris, y_iris)
    
    print("--- Iris Binary Classification Results ---")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", conf_mat)
    # print("Classification Report:\n", report) 