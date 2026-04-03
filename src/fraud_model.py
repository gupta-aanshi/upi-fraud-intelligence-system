import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay
)
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import joblib
import os

# Loading data 
df = pd.read_csv("data/transactions.csv")

# Feature Engineering (15+ features)
df = df.sort_values("transaction_id")

# Time features
df["hour"] = df["hour"].astype(int)
df["is_night"] = df["hour"].apply(lambda h: 1 if h < 6 or h >= 22 else 0)
df["is_peak"] = df["hour"].apply(lambda h: 1 if 9 <= h <= 11 or 19 <= h <= 21 else 0)

# Amount features
df["amount_zscore"] = (df["amount"] - df["amount"].mean()) / df["amount"].std()
df["is_round_amount"] = (df["amount"] % 100 == 0).astype(int)
df["amount_log"] = np.log1p(df["amount"])

# Sender velocity features (transactions per sender)
sender_counts = df.groupby("sender_id")["transaction_id"].transform("count")
df["sender_txn_count"] = sender_counts

sender_avg_amount = df.groupby("sender_id")["amount"].transform("mean")
df["sender_avg_amount"] = sender_avg_amount
df["amount_vs_sender_avg"] = df["amount"] / (df["sender_avg_amount"] + 1)

# Recipient frequency
receiver_counts = df.groupby("receiver_id")["transaction_id"].transform("count")
df["receiver_txn_count"] = receiver_counts

# Device type encoding
df["device_encoded"] = df["device"].astype("category").cat.codes

# State encoding
df["state_encoded"] = df["state"].astype("category").cat.codes

# Sender–receiver pair frequency
pair_counts = df.groupby(["sender_id", "receiver_id"])["transaction_id"].transform("count")
df["pair_frequency"] = pair_counts

# Is new receiver for this sender?
df["is_new_receiver"] = (df["pair_frequency"] == 1).astype(int)

FEATURES = [
    "amount", "amount_log", "amount_zscore", "is_round_amount",
    "hour", "is_night", "is_peak",
    "sender_txn_count", "sender_avg_amount", "amount_vs_sender_avg",
    "receiver_txn_count", "pair_frequency", "is_new_receiver",
    "device_encoded", "state_encoded"
]

X = df[FEATURES]
y = df["fraud_flag"]

print(f"Class distribution:\n{y.value_counts()}")
print(f"Fraud rate: {y.mean():.2%}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE — handle 3% class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")

# XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,  # SMOTE is already balanced
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_res, y_train_res,
          eval_set=[(X_test, y_test)],
          verbose=50)

# Full Evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

print(f"ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")
print(f"PR-AUC:   {average_precision_score(y_test, y_proba):.4f}")

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Ploting ROC + PR curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[0], name="XGBoost")
axes[0].set_title("ROC Curve")
PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axes[1], name="XGBoost")
axes[1].set_title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig("images/model_evaluation.png", dpi=150)
print("Saved: images/model_evaluation.png")

# Baseline Comparison Table
from sklearn.ensemble import IsolationForest
from sklearn.dummy import DummyClassifier

baselines = {}

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
baselines["Threshold (majority class)"] = {
    "ROC-AUC": roc_auc_score(y_test, dummy.predict_proba(X_test)[:, 1]),
    "PR-AUC": average_precision_score(y_test, dummy.predict_proba(X_test)[:, 1])
}

iso = IsolationForest(contamination=0.03, random_state=42)
iso.fit(X_train[["amount", "hour"]])  # original 2 features
iso_scores = -iso.score_samples(X_test[["amount", "hour"]])
baselines["Isolation Forest (original)"] = {
    "ROC-AUC": roc_auc_score(y_test, iso_scores),
    "PR-AUC": average_precision_score(y_test, iso_scores)
}

baselines["XGBoost + SMOTE (ours)"] = {
    "ROC-AUC": roc_auc_score(y_test, y_proba),
    "PR-AUC": average_precision_score(y_test, y_proba)
}

print("\n=== Baseline Comparison ===")
print(pd.DataFrame(baselines).T.to_string())

# SHAP Explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:500])  # sample for speed

plt.figure()
shap.summary_plot(shap_values, X_test[:500], feature_names=FEATURES,
                  show=False, max_display=15)
plt.tight_layout()
plt.savefig("images/shap_importance.png", dpi=150, bbox_inches="tight")
print("Saved: images/shap_importance.png")

# Saving model + enriched data
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgboost_fraud.pkl")
joblib.dump(FEATURES, "models/feature_list.pkl")

df["xgb_fraud_score"] = model.predict_proba(X[FEATURES])[:, 1]
df["xgb_fraud_flag"] = model.predict(X[FEATURES])
df.to_csv("data/transactions_with_anomalies.csv", index=False)
print("Done! Model saved to models/xgboost_fraud.pkl")


# fraud score distribution
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

legit_scores = df[df["fraud_flag"] == 0]["xgb_fraud_score"]
fraud_scores = df[df["fraud_flag"] == 1]["xgb_fraud_score"]

plt.hist(legit_scores, bins=30, alpha=0.6, label="Legit", density=True)
plt.hist(fraud_scores, bins=30, alpha=0.6, label="Fraud", density=True)

plt.xlim(0,1)
plt.xlabel("Fraud Score")
plt.ylabel("Density")
plt.title("Fraud Score Distribution")
plt.legend()

plt.savefig("images/fraud_score_distribution.png", dpi=200)

