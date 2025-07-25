import sys
import joblib
import numpy as np
sys.path.append("../src")
from model import FraudModel

def test_model_prediction():
    # Load processed data
    X = np.loadtxt("../data/processed/X_test.csv", delimiter=",")
    y = np.loadtxt("../data/processed/y_test.csv", delimiter=",")

    # Load saved model
    model = joblib.load("../scripts/tuned_model.pkl")

    # Predict on a small sample
    preds = model.predict(X[:5])
    assert len(preds) == 5
