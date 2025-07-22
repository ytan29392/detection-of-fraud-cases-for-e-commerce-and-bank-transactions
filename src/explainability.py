import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd

class ModelExplainability:
    def __init__(self, model_path="../scripts/fraud_model.pkl"):
        self.model = joblib.load(model_path)
        self.explainer = None

    def compute_shap_values(self, X_sample):
        """Create SHAP explainer and compute SHAP values for a sample dataset."""
        self.explainer = shap.TreeExplainer(self.model)
        shap_values = self.explainer.shap_values(X_sample)
        return shap_values

    def plot_summary(self, shap_values, X_sample, feature_names=None):
        """Global feature importance summary plot."""
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names)

    def plot_force(self, shap_values, X_sample, index=0):
        """Explain a single prediction (local interpretability)."""
        shap.initjs()
        shap.force_plot(
            self.explainer.expected_value[1],  # For the fraud (class 1) side
            shap_values[1][index],
            X_sample[index],
            matplotlib=True
        )
        plt.show()
