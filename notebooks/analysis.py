import pandas as pd

# load dataset
df = pd.read_csv("data/transactions.csv")

print("Dataset loaded successfully\n")

# show first 5 rows
print(df.head())

# counting total transactions
print("\nTotal transactions:", len(df))

# counting fraud transactions
fraud_count = df["fraud_flag"].sum()

print("Fraud transactions:", fraud_count)   # out of 50,000, 1517 transactions are fraud i.e. = 3 % transactions are fake

# finding fraud transactions by state = thi answers which state has the most fraud and according to my data it is -> BIHAR
fraud_by_state = df[df["fraud_flag"] == 1]["state"].value_counts()

print("\nFraud by state:")
print(fraud_by_state)
 
# fraud by time of day = at hour 21 fraud happens the most
fraud_by_hour = df[df["fraud_flag"] == 1]["hour"].value_counts().sort_index()

print("\nFraud by hour:")
print(fraud_by_hour)



# VISUALISATION
import matplotlib.pyplot as plt
import seaborn as sns

# Fraud by state chart
plt.figure(figsize=(8,5))
sns.barplot(x=fraud_by_state.index, y=fraud_by_state.values)

plt.title("Fraud Transactions by State")
plt.xlabel("State")
plt.ylabel("Number of Fraud Transactions")

plt.show()