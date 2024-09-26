import pandas as pd

def load_data():
    data = pd.read_csv("iris.csv")
    X = data.drop("species", axis=1)
    y = data["species"]
    return X, y