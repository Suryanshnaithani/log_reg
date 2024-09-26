import pytest
from src import model, data
@pytest.mark.parametrize
def test_model_accuracy():
    X, y = data.load_data()
    model = model.train_model(X, y)
    accuracy = model.score(X, y)
    assert accuracy > 0.8  # Adjust threshold as needed