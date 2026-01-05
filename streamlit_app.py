import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8080")

st.set_page_config(page_title="Churn Assistant", page_icon="📊")

st.title("📊 Churn Prediction Assistant")
st.write("Ask about a customer in plain language, and get churn risk + recommendations.")

# Sidebar info
st.sidebar.header("Settings")
st.sidebar.write("This app uses:")
st.sidebar.write("- Logistic Regression churn model")
st.sidebar.write("- Groq-hosted LLM for conversation parsing")
st.sidebar.write("- FastAPI backend running at:", API_URL)

# Tabs
tab1, tab2 = st.tabs(["Chatbot", "Direct Prediction"])

# --- Tab 1: Chatbot ---
with tab1:
    st.subheader("💬 Chat with the assistant")
    user_msg = st.text_area("Enter customer details (natural language):", height=100)

    if st.button("Analyze via Chatbot"):
        if not user_msg.strip():
            st.warning("Please enter some customer details.")
        else:
            try:
                resp = requests.post(f"{API_URL}/chat", json={"message": user_msg}, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                if "churn_probability" in data:
                    st.success(f"Churn probability: {data['churn_probability']:.2f}")
                    st.write("### Predicted Label")
                    st.info(f"{data['churn_label']}")
                    st.write("### Insights / Recommended Actions")
                    for rec in data["insights"]:
                        st.markdown(f"- {rec}")

                    st.write("### Extracted Features")
                    st.json(data["extracted_features"])
                else:
                    st.info(data.get("response", "No valid prediction available."))

            except Exception as e:
                st.error(f"Error: {e}")

# --- Tab 2: Direct Prediction ---
with tab2:
    st.subheader("📝 Direct Form Input")

    with st.form("predict_form"):
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Credit card (automatic)", "Bank transfer (automatic)"]
        )
        Monthly_Charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)

        submitted = st.form_submit_button("Predict")

        if submitted:
            payload = {
                "SeniorCitizen": SeniorCitizen,
                "Partner": Partner,
                "Dependents": Dependents,
                "tenure": tenure,
                "InternetService": InternetService,
                "OnlineSecurity": OnlineSecurity,
                "OnlineBackup": OnlineBackup,
                "DeviceProtection": DeviceProtection,
                "TechSupport": TechSupport,
                "Contract": Contract,
                "PaperlessBilling": PaperlessBilling,
                "PaymentMethod": PaymentMethod,
                "Monthly_Charges": Monthly_Charges
            }

            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                st.success(f"Churn probability: {data['churn_probability']:.2f}")
                st.info(f"Predicted Label: {data['churn_label']}")
            except Exception as e:
                st.error(f"Error: {e}")
