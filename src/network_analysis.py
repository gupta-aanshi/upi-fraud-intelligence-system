import matplotlib
matplotlib.use('TkAgg')


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv("data/transactions_with_anomalies.csv")

print("Dataset loaded")
print(df.head())

# creating sender and receiver accounts 
df["sender"] = "U" + df["transaction_id"].astype(str)
df["receiver"] = "U" + (df["transaction_id"] + 1).astype(str)

# creating the network graph
G = nx.from_pandas_edgelist(df, "sender", "receiver")

# displaying basic network stats
print("\nTotal users:", G.number_of_nodes())
print("Total transactions:", G.number_of_edges())

# VISUALISATION
# take only first 200 transactions for visualization
sample_df = df.head(200)

G = nx.from_pandas_edgelist(sample_df, "sender", "receiver")

print("Total users:", G.number_of_nodes())
print("Total transactions:", G.number_of_edges())

plt.figure(figsize=(10,7))

nx.draw(G, node_size=100, with_labels=False)

plt.title("Sample UPI Transaction Network")

plt.show()