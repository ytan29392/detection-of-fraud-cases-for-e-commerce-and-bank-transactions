import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import os

class DataPipeline:
    def __init__(self, filepath="../data/Fraud_Data.csv", target_col="is_fraud"):
        self.filepath = filepath
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)

    def load_data(self):
        """Load dataset from CSV."""
        df = pd.read_csv(self.filepath)
        return df

    def preprocess(self, df):
        """Clean dataset: fill missing values, encode categoricals."""
        df.fillna(0, inplace=True)

        # One-hot encode categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        return df

    def balance_and_scale(self, df):
        """Balance fraud vs non-fraud using SMOTE and scale features."""
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # Apply SMOTE
        X_res, y_res = self.smote.fit_resample(X, y)

        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X_res)
        return X_scaled, y_res

    def split_data(self, X, y, test_size=0.2):
        """Split dataset into train and test sets (stratified)."""
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    def save_splits(self, X_train, X_test, y_train, y_test, output_dir="../data/processed"):
        """Save processed train/test splits to CSV."""
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(X_train).to_csv(f"{output_dir}/X_train.csv", index=False)
        pd.DataFrame(X_test).to_csv(f"{output_dir}/X_test.csv", index=False)
        pd.DataFrame(y_train).to_csv(f"{output_dir}/y_train.csv", index=False)
        pd.DataFrame(y_test).to_csv(f"{output_dir}/y_test.csv", index=False)