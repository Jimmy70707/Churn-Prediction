from fastapi import FastAPI
import pandas as pd
import joblib
from app import chatbot_llm  # import our sync LLM wrapper

# Load model, encoders, and feature order from .pkl
model, encoders, feature_order = joblib.load("log_reg_with_encoders.pkl")
feature_order = [col.strip() for col in feature_order]

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn from provided customer data or via chat",
    version="1.3"
)

def preprocess_input(customer_data: dict) -> pd.DataFrame:
    df = pd.DataFrame([customer_data])
    df.columns = df.columns.str.strip()

    for col in feature_order:
        if col not in df.columns:
            df[col] = encoders[col].classes_[0] if col in encoders else 0

    for col, le in encoders.items():
        if col in df:
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))

    return df[feature_order]

def generate_insights(customer_data: dict, churn_label: str, churn_prob: float) -> list:
    insights = []

    if customer_data.get("Contract") == "Month-to-month":
        insights.append("Month-to-month contracts are associated with higher churn risk.")
    else:
        insights.append("Longer contracts usually reduce churn risk.")

    if customer_data.get("tenure", 0) < 12:
        insights.append("Customers with tenure below 12 months are more likely to churn.")
    else:
        insights.append("Stable customers with longer tenure are less likely to churn.")

    if customer_data.get("Monthly_Charges", 0) > 80:
        insights.append("High monthly charges may increase churn risk.")
    else:
        insights.append("Affordable monthly charges can help reduce churn risk.")

    if customer_data.get("SeniorCitizen", 0) == 1:
        insights.append("Senior citizens may be more sensitive to service value.")

    if churn_label == "Yes":
        insights.append(f"The model predicts churn with probability {churn_prob:.2f}. Key factors above may contribute.")
    else:
        insights.append(f"The model predicts no churn with probability {churn_prob:.2f}. Profile appears stable.")

    return insights

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running!"}

@app.post("/predict")
def predict_churn(customer_data: dict):
    try:
        df_processed = preprocess_input(customer_data)
        pred = model.predict(df_processed)[0]
        prob = model.predict_proba(df_processed)[0][1]
        churn_label = "Yes" if pred == 1 else "No"

        return {
            "churn_label": churn_label,
            "churn_probability": round(float(prob), 4),
            "insights": generate_insights(customer_data, churn_label, prob)
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
def chat_with_model(payload: dict):
    try:
        message = payload.get("message", "")
        extracted_json = chatbot_llm.natural_to_json(message)

        if isinstance(extracted_json, dict) and extracted_json:
            df_processed = preprocess_input(extracted_json)
            pred = model.predict(df_processed)[0]
            prob = model.predict_proba(df_processed)[0][1]
            churn_label = "Yes" if pred == 1 else "No"

            return {
                "churn_label": churn_label,
                "churn_probability": round(float(prob), 4),
                "extracted_features": extracted_json,
                "insights": generate_insights(extracted_json, churn_label, prob)
            }
        else:
            return {"response": "Could not extract customer details. Please try again with more structured info."}

    except Exception as e:
        return {"error": str(e)}
