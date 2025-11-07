# algorithms/dt_wine.py
"""
Decision Tree on Wine Dataset containing 3 Classes, 13 Features
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_wine

def load_wine_data():
    """Loads the Wine dataset."""
    data = load_wine()
    return data.data, data.target, data.target_names

def train_wine_dt_model(X, y, test_size=0.3, random_state=42):
    """Splits data and trains a Decision Tree model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_wine_model(model, X_test, y_test, target_names):
    """Evaluates the model and returns accuracy and classification report."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, target_names=target_names)
    
    return accuracy, report
if __name__ == '__main__':
    # Load the data
    X, y, target_names = load_wine_data()
    
    # Train the model and get test data
    model, X_test, y_test = train_wine_dt_model(X, y, test_size=0.3)
    
    # Evaluate the model
    accuracy, report = evaluate_wine_model(model, X_test, y_test, target_names)
    
    # Print the result to the terminal
    print("\n--- Wine Data Decision Tree Execution ---")
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print("Classification Report Summary:")
    for class_name, metrics in report.items():
        if class_name in target_names:
            print(f"Class: {class_name}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1-score']:.4f}")