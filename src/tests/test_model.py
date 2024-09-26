# tests/test_model.py
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.model import load_data, train_model, evaluate_model

# Test loading of data
def test_load_data():
    X, y = load_data()
    assert X.shape == (150, 4), "Feature matrix should have 150 samples and 4 features"
    assert len(y) == 150, "Target vector should have 150 samples"

# Test training the model
def test_train_model():
    X, y = load_data()
    model, X_test, y_test = train_model(X, y)
    assert isinstance(model, LogisticRegression), "The model should be a LogisticRegression instance"
    assert X_test.shape[1] == 4, "The test set should have 4 features"
    assert len(y_test) > 0, "The test set should not be empty"

# Test model evaluation
def test_evaluate_model():
    X, y = load_data()
    model, X_test, y_test = train_model(X, y)
    accuracy = evaluate_model(model, X_test, y_test)
    assert accuracy > 0.9, "Model accuracy should be greater than 90%"
