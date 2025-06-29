import joblib
import numpy as np
from sklearn.metrics import accuracy_score

def test_model_accuracy_threshold():
    model = joblib.load("model.joblib")
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")
    acc = accuracy_score(y_test, model.predict(X_test))
    assert acc > 0.85, f"Expected accuracy > 0.85, got {acc}"
