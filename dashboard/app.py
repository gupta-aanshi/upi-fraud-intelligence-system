import streamlit as st
import pandas as pd

# Load dataset
df = pd.read_csv("data/transactions_with_anomalies.csv")

st.title("UPI Fraud Intelligence Dashboard")

st.write("### Dataset Overview")

st.write("Total Transactions:", len(df))
st.write("Fraud Transactions:", df["fraud_flag"].sum())

# Fraud by state
st.write("### Fraud by State")

fraud_by_state = df[df["fraud_flag"] == 1]["state"].value_counts()

st.bar_chart(fraud_by_state)

# Fraud by hour
st.write("### Fraud by Hour")

fraud_by_hour = df[df["fraud_flag"] == 1]["hour"].value_counts().sort_index()

st.line_chart(fraud_by_hour)

# Suspicious transactions
st.write("### Suspicious Transactions (Detected by ML)")

suspicious = df[df["anomaly"] == -1]

st.write(suspicious.head(10))