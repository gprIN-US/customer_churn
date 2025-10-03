import os
import pandas as pd
import requests
import streamlit as st
from pathlib import Path

API_URL = os.environ.get("API_URL", "http://localhost:8787")
DATA = Path(__file__).resolve().parents[2] / "data" / "customers.csv"

st.set_page_config(page_title="Churn Early Warning", layout="wide")

st.title("Customer Churn Early Warning — Business View")
st.caption("Tune risk threshold and preview who gets contacted.")

with st.sidebar:
    st.header("Settings")
    thr = st.slider("Risk Threshold", min_value=0.05, max_value=0.95, value=0.5, step=0.05)
    st.write("API:", API_URL)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload or use demo data")
    uploaded = st.file_uploader("CSV with feature columns", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv(DATA)
    st.write("Rows:", len(df))
    st.dataframe(df.head(10))

with col2:
    st.subheader("Batch score preview (first 50)")
    df50 = df.head(50).copy()
    scores = []
    for _, r in df50.iterrows():
        payload = {
            "monthly_charges": float(r["monthly_charges"]),
            "tenure_months": int(r["tenure_months"]),
            "is_promo": int(r["is_promo"]),
            "support_tickets_90d": int(r["support_tickets_90d"]),
            "late_payments_12m": int(r["late_payments_12m"]),
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=2).json()
            scores.append(resp["risk_score"])
        except Exception:
            scores.append(0.0)
    df50["risk_score"] = scores
    df50["contact" ] = (df50["risk_score"] >= thr)
    st.write("Contact list preview:")
    st.dataframe(df50[["monthly_charges","tenure_months","risk_score","contact"]])

st.markdown("---")
st.subheader("Business guidance")
st.markdown(
    "- Use a lower threshold to maximize recall of churners (more contacts, more cost).\n"
    "- Use a higher threshold to increase precision (fewer contacts, higher hit-rate).\n"
    "- Always A/B test contact scripts; the model only prioritizes - not persuades."
)
