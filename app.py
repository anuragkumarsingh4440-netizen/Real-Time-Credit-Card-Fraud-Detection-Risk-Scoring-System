# =========================================================
# AI FRAUD DETECTION + EXPLAINABLE AI SYSTEM (FINAL)
# =========================================================

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Fraud Detection AI", layout="wide")

# =========================================================
# UI STYLING
# =========================================================
st.markdown("""
<style>
.main-title {
    background: linear-gradient(90deg, #000000, #8B0000);
    color: white;
    padding: 18px;
    border-radius: 10px;
    font-size: 34px;
    font-weight: 900;
    text-align: center;
}
.kpi {
    background: #111;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">AI Fraud Detection System</div>', unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
model = joblib.load('models/fraud_model.pkl')
features = joblib.load('models/features.pkl')
threshold = joblib.load('models/threshold.pkl')

# =========================================================
# INPUT SECTION
# =========================================================
st.sidebar.header("Transaction Input")

step = st.sidebar.number_input("Step", value=1.0)
amount = st.sidebar.number_input("Transaction Amount", value=1000.0)

oldbalanceOrg = st.sidebar.number_input("Sender Balance Before", value=100000.0)
newbalanceOrig = st.sidebar.number_input("Sender Balance After", value=90000.0)

oldbalanceDest = st.sidebar.number_input("Receiver Balance Before", value=5000.0)
newbalanceDest = st.sidebar.number_input("Receiver Balance After", value=6000.0)

txn_type = st.sidebar.selectbox(
    "Transaction Type",
    ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
)

# =========================================================
# FEATURE ENGINEERING
# =========================================================
balance_diff_orig = oldbalanceOrg - newbalanceOrig
balance_diff_dest = newbalanceDest - oldbalanceDest

type_dict = {
    "type_CASH_OUT": 0,
    "type_DEBIT": 0,
    "type_PAYMENT": 0,
    "type_TRANSFER": 0
}
type_dict[f"type_{txn_type}"] = 1

input_df = pd.DataFrame([{
    "step": step,
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest,
    "isFlaggedFraud": 0,
    "balance_diff_orig": balance_diff_orig,
    "balance_diff_dest": balance_diff_dest,
    **type_dict
}])

# Ensure feature alignment
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[features]

# =========================================================
# PREDICTION
# =========================================================
prob = model.predict_proba(input_df)[:, 1]
pred = (prob > threshold).astype(int)

# =========================================================
# KPI DISPLAY
# =========================================================
st.subheader("Fraud Result")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Fraud Probability", f"{prob[0]:.2f}")

with c2:
    label = "🚨 FRAUD" if pred[0] == 1 else "✅ SAFE"
    st.metric("Prediction", label)

with c3:
    risk = "HIGH" if prob[0] > 0.7 else "MEDIUM" if prob[0] > 0.4 else "LOW"
    st.metric("Risk Level", risk)
if oldbalanceOrg > 0 and newbalanceOrig == 0 and amount == oldbalanceOrg:
    st.error("🚨 High Risk Pattern Detected: Full Balance Drain")

# =========================================================
# SHAP (REAL-TIME INSTANCE LEVEL)
# =========================================================
explainer = shap.TreeExplainer(model)
shap_values = explainer(input_df)

shap_df = pd.DataFrame({
    "Feature": input_df.columns,
    "Impact": shap_values.values[0]
}).sort_values(by="Impact", key=abs, ascending=False)

st.subheader("Top Factors Affecting This Transaction")
st.dataframe(shap_df.head(10))

# =========================================================
# PARALLEL VISUALS
# =========================================================
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(4,3))
    ax.barh(shap_df["Feature"][:8], shap_df["Impact"][:8])
    ax.invert_yaxis()
    st.pyplot(fig)

with col2:
    fig2, ax2 = plt.subplots(figsize=(4,3))
    ax2.hist(prob, bins=10)
    st.pyplot(fig2)

# =========================================================
# GEMINI AI EXPLANATION
# =========================================================
st.subheader("AI Fraud Explanation")

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if st.button("Generate Explanation"):

    if api_key:

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)

            model_ai = genai.GenerativeModel("gemini-2.5-flash-lite")

            top_features = shap_df.head(5).to_string(index=False)

            prompt = f"""
You are a fraud detection expert.

Fraud Probability: {prob[0]:.2f}
Prediction: {"FRAUD" if pred[0]==1 else "SAFE"}

Top Factors:
{top_features}

Explain:
- Why this transaction is fraud or safe
- What is suspicious
- What user should do

Use simple bullet points.
"""

            response = model_ai.generate_content(prompt)

            st.success("AI Explanation")
            st.write(response.text)

        except Exception as e:
            st.error(f"AI Error: {e}")

    else:
        st.warning("API Key missing. Check .env file.")