import pandas as pd
from sklearn.ensemble import IsolationForest

# load dataset
df = pd.read_csv("data/transactions.csv")

print("Dataset loaded")
print(df.head())

# Selecting features for our ML model = unusual large amount and unusual time
features = df[["amount", "hour"]]

# creating isolation forest model
model = IsolationForest(contamination=0.03, random_state=42)   # contamination = expected fraud rate (3%)

# training the model
df["anomaly"] = model.fit_predict(features)   # 1 = normal transaction | -1 - suspicious transaction


# displaying detected suspicious transactions
suspicious = df[df["anomaly"] == -1]

print("\nSuspicious transactions detected:")
print(suspicious.head())

print("\nTotal suspicious detected:", len(suspicious))

# saving the ML results
df.to_csv("data/transactions_with_anomalies.csv", index=False)

print("\nDataset with anomaly results saved")