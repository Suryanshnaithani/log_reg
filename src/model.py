
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Function to load data
def load_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Function to train the logistic regression model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # Save the model to a file
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model, X_test, y_test

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
