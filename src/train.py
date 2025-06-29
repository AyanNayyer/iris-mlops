import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("iris.csv")

# Preprocess
X = df.drop(columns=["species"])
y = df[["species"]]

encoder = OrdinalEncoder()
y_encoded = encoder.fit_transform(y).ravel()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# Save model and encoder
joblib.dump(model, "model.joblib")
joblib.dump(encoder, "encoder.joblib")

# Save test set for metrics plotting
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
np.save("y_pred.npy", y_pred)
