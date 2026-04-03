import streamlit as st
import requests

st.title("UPI Fraud Detection - Live Scoring")

amount = st.number_input("Transaction Amount", value=500)
hour = st.slider("Hour", 0, 23, 12)
device = st.selectbox("Device", ["Android", "iOS", "Web"])
state = st.selectbox("State", ["UP","Delhi","Bihar","Maharashtra"])

if st.button("Check Fraud Risk"):

    payload = {
        "amount": amount,
        "hour": hour,
        "sender_id": "user_0001",
        "receiver_id": "user_0400",
        "device": device,
        "state": state
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=payload
    )

    result = response.json()

    st.write("### Fraud Score:", result["fraud_score"])
    st.write("### Risk Level:", result["risk_level"])
    st.write("### Latency (ms):", result["latency_ms"])