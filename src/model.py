# src/model.py

import joblib
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class FraudModel:
    def __init__(self):
        # Initialize models we want to compare
        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False)
        }
        self.trained_models = {}

    def train(self, X_train, y_train):
        """Train all models and store them."""
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model

    def evaluate(self, X_test, y_test):
        """Evaluate all trained models and return scores."""
        scores = {}
        for name, model in self.trained_models.items():
            preds = model.predict(X_test)
            report = classification_report(y_test, preds, output_dict=True)
            roc_auc = roc_auc_score(y_test, preds)
            scores[name] = {
                "precision": report["1"]["precision"],
                "recall": report["1"]["recall"],
                "f1-score": report["1"]["f1-score"],
                "roc-auc": roc_auc
            }
            print(f"\nModel: {name}")
            print(classification_report(y_test, preds))
            print("ROC-AUC:", roc_auc)
        return scores

    def save_model(self, model_name, path="../scripts/fraud_model.pkl"):
        """Save selected model to disk."""
        if model_name in self.trained_models:
            joblib.dump(self.trained_models[model_name], path)
            print(f"Model '{model_name}' saved to {path}")
        else:
            print(f"Model '{model_name}' not found. Train first!")