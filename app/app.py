import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# -------------------- Load Model --------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "telecom_model.pkl")
model = joblib.load(model_path)

# -------------------- Page Config --------------------
st.set_page_config(page_title="Telecom AI Dashboard", layout="wide")

st.markdown("# 📊 Telecom AI Dashboard")
st.markdown("### Predict customer monthly charges using Machine Learning")

# -------------------- Layout --------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Info")
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    Partner = st.selectbox("Partner", ["No", "Yes"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    st.subheader("📡 Services")
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes"])

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    st.subheader("💳 Billing")
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    )

with col4:
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# -------------------- Encoding --------------------
def encode(df):
    for col in df.columns:
        df[col] = df[col].astype("category").cat.codes
    return df

# -------------------- Prediction --------------------
st.markdown("## 🔮 Prediction")

if st.button("Predict Monthly Charges"):
    data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "TotalCharges": TotalCharges
    }])

    data = encode(data)
    prediction = model.predict(data)[0]

    st.success(f"💰 Estimated Monthly Charges: ₹{prediction:.2f}")

    # -------------------- Chart --------------------
    st.subheader("📊 Prediction Visualization")

    chart_data = pd.DataFrame({
        "Category": ["Predicted Charges"],
        "Value": [prediction]
    })

    st.bar_chart(chart_data.set_index("Category"))

# -------------------- Analytics Section --------------------
st.markdown("---")
st.markdown("## 📈 Analytics")

# Dummy data (replace with real dataset later)
analytics_data = pd.DataFrame({
    "Month": range(1, 13),
    "Avg Charges": [100,120,150,180,200,220,250,270,300,320,350,370]
})

st.line_chart(analytics_data.set_index("Month"))

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Built with ❤️ using Machine Learning & Streamlit")