import pickle
from src import model, data

def deploy_model():
    X, y = data.load_data()
    model = model.train_model(X, y)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
