from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import time

app = FastAPI(
    title="UPI Fraud Detection API",
    description="Real-time UPI transaction fraud scoring in <50ms",
    version="1.0.0"
)

# Loading model on startup
model = joblib.load("models/xgboost_fraud.pkl")
FEATURES = joblib.load("models/feature_list.pkl")

# Loading training stats for zscore calculation
df = pd.read_csv("data/transactions.csv")
AMOUNT_MEAN = df["amount"].mean()
AMOUNT_STD = df["amount"].std()

class Transaction(BaseModel):
    amount: float
    hour: int
    sender_id: str
    receiver_id: str
    device: str
    state: str
    sender_txn_count: int = 5
    sender_avg_amount: float = 800.0
    receiver_txn_count: int = 3
    pair_frequency: int = 1

class FraudPrediction(BaseModel):
    transaction_id: str
    fraud_score: float
    is_fraud: bool
    risk_level: str
    latency_ms: float

def engineer_features(txn: Transaction) -> pd.DataFrame:
    amount = txn.amount
    hour = txn.hour
    row = {
        "amount": amount,
        "amount_log": np.log1p(amount),
        "amount_zscore": (amount - AMOUNT_MEAN) / AMOUNT_STD,
        "is_round_amount": int(amount % 100 == 0),
        "hour": hour,
        "is_night": int(hour < 6 or hour >= 22),
        "is_peak": int(9 <= hour <= 11 or 19 <= hour <= 21),
        "sender_txn_count": txn.sender_txn_count,
        "sender_avg_amount": txn.sender_avg_amount,
        "amount_vs_sender_avg": amount / (txn.sender_avg_amount + 1),
        "receiver_txn_count": txn.receiver_txn_count,
        "pair_frequency": txn.pair_frequency,
        "is_new_receiver": int(txn.pair_frequency == 1),
        "device_encoded": {"Android": 0, "iOS": 1, "Web": 2}.get(txn.device, 0),
        "state_encoded": hash(txn.state) % 30,
    }
    return pd.DataFrame([row])[FEATURES]

@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost_fraud_v1"}

@app.post("/predict", response_model=FraudPrediction)
def predict(txn: Transaction):
    start = time.time()
    try:
        X = engineer_features(txn)
        score = float(model.predict_proba(X)[0][1])
        is_fraud = score > 0.5
        risk = "HIGH" if score > 0.7 else "MEDIUM" if score > 0.4 else "LOW"
        latency = (time.time() - start) * 1000
        return FraudPrediction(
            transaction_id=f"txn_{int(time.time()*1000)}",
            fraud_score=round(score, 4),
            is_fraud=is_fraud,
            risk_level=risk,
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))