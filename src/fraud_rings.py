import pandas as pd
import networkx as nx

df = pd.read_csv("data/transactions_with_anomalies.csv")

fraud_df = df[df["xgb_fraud_flag"] == 1]

G = nx.from_pandas_edgelist(
    fraud_df,
    "sender_id",
    "receiver_id"
)

components = list(nx.connected_components(G))

rings = [c for c in components if len(c) > 3]

print("Fraud Rings Detected:", len(rings))