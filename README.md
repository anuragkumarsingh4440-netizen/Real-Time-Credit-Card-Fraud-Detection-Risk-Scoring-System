# 💳 AI-Powered Real-Time Fraud Detection & Explainability System

## 📌 Overview
This project is an end-to-end **Fraud Detection System** designed to identify suspicious financial transactions in real time using Machine Learning and Explainable AI.

The system not only predicts whether a transaction is **fraud or safe**, but also explains **why** the prediction was made using SHAP and AI-generated insights.

---

## 🎯 Key Features

- 🔍 Real-time Fraud Detection (XGBoost Model)
- 📊 Probability-based Risk Scoring
- 🧠 Explainable AI using SHAP (Feature Impact)
- 🤖 AI Explanation using Gemini API
- 📈 Interactive Streamlit Dashboard
- 📂 CSV Upload Support for Batch Predictions
- ⚡ Dynamic Threshold-based Decision System

---

## 🧠 Problem Statement

Financial fraud is increasing rapidly, and traditional systems lack transparency.

This system solves:
- Detecting fraud in real-time
- Explaining model decisions
- Providing actionable insights to users

---

## 🏗️ System Architecture

1. Data Preprocessing & Feature Engineering  
2. Model Training (XGBoost)  
3. Threshold Optimization  
4. SHAP Explainability  
5. Streamlit UI Deployment  
6. AI Explanation Layer (Gemini)

---

## ⚙️ Tech Stack

| Category | Tools |
|----------|------|
| Language | Python |
| ML Model | XGBoost |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib |
| Explainability | SHAP |
| UI | Streamlit |
| AI Explanation | Google Gemini API |

---

## 📊 Model Performance (After Fixing Data Leakage)

- Accuracy: ~99%
- Recall: High (fraud detection sensitive)
- Precision: Tuned via threshold optimization
- ROC-AUC: ~0.99

⚠️ Note: Data leakage (e.g., `isFlaggedFraud`) was removed to ensure realistic performance.

---

## 🧪 How It Works

### Step 1: User Input
- Transaction amount
- Sender & receiver balances
- Transaction type

### Step 2: Model Prediction
- Fraud probability is calculated
- Threshold applied for classification

### Step 3: Explainability
- SHAP shows feature contribution
- Top factors affecting decision

### Step 4: AI Explanation
- Gemini API converts model output into human-readable insights

---

## 🚀 Installation & Setup

```bash
git clone https://github.com/your-username/fraud-detection-system.git
cd fraud-detection-system
pip install -r requirements.txt
streamlit run app.py