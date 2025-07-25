import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import shap

# Load the tuned model
MODEL_PATH = "../scripts/tuned_model.pkl"
model = joblib.load(MODEL_PATH)

# Initialize FastAPI
app = FastAPI(title="Fraud Detection API", version="1.0")

# For explainability
explainer = shap.TreeExplainer(model)

# Define input schema (adjust fields based on your preprocessed data shape)
class TransactionData(BaseModel):
    features: list  # List of numerical features after preprocessing

@app.post("/predict")
def predict(data: TransactionData):
    """Return fraud prediction and probability."""
    X = np.array(data.features).reshape(1, -1)
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])  # Probability of fraud
    return {"prediction": prediction, "fraud_probability": probability}

@app.post("/explain")
def explain(data: TransactionData):
    """Return SHAP values for feature contributions."""
    X = np.array(data.features).reshape(1, -1)
    shap_values = explainer.shap_values(X)
    # Return contributions for the fraud class (class 1)
    return {"shap_values": shap_values[1].tolist()}
