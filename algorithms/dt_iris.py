# algorithms/dt_iris.py
"""
Decision Tree on Iris Dataset 
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

def load_iris_data():
    """Loads the Iris dataset."""
    data = load_iris()
    return data.data, data.target

def train_iris_dt_model(X, y, test_size=0.3, random_state=42):
    """Splits data and trains a Decision Tree model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_iris_model(model, X_test, y_test, target_names):
    """Evaluates the model and returns accuracy and classification report."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, target_names=target_names)
    
    return accuracy, report

if __name__ == '__main__':
    # Load the data
    X, y = load_iris_data()
    target_names = load_iris().target_names
    
    # Train the model and get test data
    model, X_test, y_test = train_iris_dt_model(X, y, test_size=0.3)
    
    # Evaluate the model
    accuracy, report = evaluate_iris_model(model, X_test, y_test, target_names)

    # Print the result to the terminal
    print("\n Iris Data Decision Tree Execution ")
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print("Classification Report Summary:")