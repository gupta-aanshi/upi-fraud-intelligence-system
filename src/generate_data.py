import pandas as pd
import numpy as np

n = 50000
np.random.seed(42)

states = ["UP", "Delhi", "Maharashtra", "Karnataka", "Bihar",
          "Rajasthan", "Gujarat", "Tamil Nadu", "West Bengal", "Punjab"]

user_ids = [f"user_{i:04d}" for i in range(500)]

# ── Generate LEGIT transactions (47,000) ─────────────────────
legit_n = 47000
legit = pd.DataFrame({
    "transaction_id": [f"txn_{i:06d}" for i in range(legit_n)],
    "amount": np.random.exponential(800, legit_n),      # small amounts
    "hour": np.random.randint(8, 22, legit_n),           # daytime only
    "state": np.random.choice(states, legit_n),
    "device": np.random.choice(["Android", "iOS"], legit_n, p=[0.6, 0.4]),
    "sender_id": np.random.choice(user_ids, legit_n),
    "receiver_id": np.random.choice(user_ids, legit_n),
    "fraud_flag": 0
})

# ── Generate FRAUD transactions (3,000) ──────────────────────
fraud_n = 3000
fraud = pd.DataFrame({
    "transaction_id": [f"txn_{i:06d}" for i in range(legit_n, legit_n + fraud_n)],
    "amount": np.random.exponential(8000, fraud_n),     # large amounts
    "hour": np.random.choice([0, 1, 2, 3, 4, 22, 23], fraud_n),  # late night
    "state": np.random.choice(["UP", "Bihar"], fraud_n),  # high fraud states
    "device": np.random.choice(["Web", "Android"], fraud_n, p=[0.7, 0.3]),  # mostly web
    "sender_id": np.random.choice(user_ids[:50], fraud_n),   # few senders (fraud rings)
    "receiver_id": np.random.choice(user_ids[450:], fraud_n), # few receivers
    "fraud_flag": 1
})

# ── Combine and shuffle ───────────────────────────────────────
data = pd.concat([legit, fraud], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Fix same sender/receiver
same_mask = data["sender_id"] == data["receiver_id"]
data.loc[same_mask, "receiver_id"] = np.random.choice(user_ids, same_mask.sum())

print(f"Total transactions: {len(data)}")
print(f"Fraud rate: {data['fraud_flag'].mean():.2%}")
print(f"\nLegit avg amount:  ₹{data[data['fraud_flag']==0]['amount'].mean():.0f}")
print(f"Fraud avg amount:  ₹{data[data['fraud_flag']==1]['amount'].mean():.0f}")
print(f"\nLegit avg hour:  {data[data['fraud_flag']==0]['hour'].mean():.1f}")
print(f"Fraud avg hour:  {data[data['fraud_flag']==1]['hour'].mean():.1f}")

data.to_csv("data/transactions.csv", index=False)
print("\n Dataset created with strong fraud patterns!")