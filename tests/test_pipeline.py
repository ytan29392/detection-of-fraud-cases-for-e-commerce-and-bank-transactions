import sys
import pandas as pd
sys.path.append("../src")
from data_pipeline import DataPipeline

def test_pipeline_runs():
    pipeline = DataPipeline(filepath="../data/sample.csv", target_col="is_fraud")
    df = pipeline.load_data()
    df_clean = pipeline.preprocess(df)
    X_scaled, y_res = pipeline.balance_and_scale(df_clean)
    assert X_scaled.shape[0] == y_res.shape[0]  # Balanced sizes
