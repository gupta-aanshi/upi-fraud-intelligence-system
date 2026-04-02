import pandas as pd
import numpy as np

n = 50000

states = ["UP","Delhi","Maharashtra","Karnataka","Bihar"]

data = pd.DataFrame({
"transaction_id": range(n),
"amount": np.random.exponential(2000,n),
"hour": np.random.randint(0,24,n),
"state": np.random.choice(states,n),
"device": np.random.choice(["Android","iOS"],n)
})

data["fraud_flag"] = np.random.choice([0,1],n,p=[0.97,0.03])

data.to_csv("data/transactions.csv", index=False)

print("Dataset created successfully")