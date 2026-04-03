import joblib
import pandas as pd

def test_model_load():
    model = joblib.load("models/xgboost_fraud.pkl")
    assert model is not None

def test_data_exists():
    df = pd.read_csv("data/transactions.csv")
    assert len(df) > 0