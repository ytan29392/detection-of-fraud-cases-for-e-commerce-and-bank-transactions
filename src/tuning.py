# src/tuning.py

import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class ModelTuning:
    def __init__(self):
        self.best_model = None
        self.best_params = None
        self.best_score = None

    def tune_random_forest(self, X_train, y_train):
        """Grid Search for RandomForestClassifier."""
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
        grid = GridSearchCV(
            rf, param_grid, cv=3, scoring="roc_auc", verbose=1, n_jobs=-1
        )
        grid.fit(X_train, y_train)

        self.best_model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.best_score = grid.best_score_
        return self.best_model, self.best_params, self.best_score

    def tune_xgboost(self, X_train, y_train):
        """Grid Search for XGBoost Classifier."""
        xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1]
        }
        grid = GridSearchCV(
            xgb, param_grid, cv=3, scoring="roc_auc", verbose=1, n_jobs=-1
        )
        grid.fit(X_train, y_train)

        self.best_model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.best_score = grid.best_score_
        return self.best_model, self.best_params, self.best_score

    def save_best_model(self, path="../scripts/tuned_model.pkl"):
        """Save the best model found."""
        if self.best_model:
            joblib.dump(self.best_model, path)
            print(f"Best tuned model saved at {path}")
        else:
            print("No tuned model to save. Run tuning first.")
