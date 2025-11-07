# logistic_regression_cancer.py
"""
Logistic Regression Model Implementation for Breast Cancer Classification
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler 

def load_cancer_data():
    """
    Loads the Breast Cancer dataset (569 samples, 30 features).
    Returns: X (features), y (labels), feature_names
    """
    data = load_breast_cancer()
    return data.data, data.target, data.feature_names

def train_and_evaluate_cancer_model(X, y, test_size=0.3, random_state=42):
    """
    Splits data, scales features, trains Logistic Regression model, and evaluates performance.
    
    Returns: model, accuracy, report, conf_matrix
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scaling features is recommended for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the Logistic Regression model
    # max_iter is increased because this dataset often requires more iterations
    model = LogisticRegression(solver='liblinear', random_state=random_state, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Prediction and Evaluation
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, target_names=['Malignant', 'Benign'])
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, report, conf_matrix

if __name__ == '__main__':
    # sample execution when run directly
    X_cancer, y_cancer, _ = load_cancer_data()
    _, acc, report, conf_mat = train_and_evaluate_cancer_model(X_cancer, y_cancer)
    
    print("--- Breast Cancer Classification Results ---")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", conf_mat)