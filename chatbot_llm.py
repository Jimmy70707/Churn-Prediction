import json
from groq import Groq

# Initialize Groq client (make sure GROQ_API_KEY is set in your environment)
client = Groq()

# Expected features for the model
EXPECTED_FEATURES = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "Monthly_Charges"
]

def natural_to_json(user_input: str) -> dict:
    """Convert natural language customer description into structured JSON."""
    prompt = f"""
You are a strict JSON generator for churn prediction.
Extract the following features from the user message, and return ONLY valid JSON with these keys:
{EXPECTED_FEATURES}

Message:
{user_input}

Rules:
- SeniorCitizen: 0 or 1
- Partner, Dependents, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, PaperlessBilling: "Yes" or "No"
- InternetService: "DSL", "Fiber optic", or "No"
- Contract: "Month-to-month", "One year", "Two year"
- PaymentMethod: "Electronic check", "Mailed check", "Credit card (automatic)", "Bank transfer (automatic)"
- Monthly_Charges: number
- tenure: integer months

Return example:
{{
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "Yes",
  "DeviceProtection": "Yes",
  "TechSupport": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "Monthly_Charges": 70.0
}}
"""
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )
        raw_output = response.choices[0].message.content.strip()

        # Force JSON extraction
        if "{" in raw_output:
            raw_output = raw_output[raw_output.index("{"):raw_output.rindex("}") + 1]

        data = json.loads(raw_output)

        # Ensure all expected keys are present
        for key in EXPECTED_FEATURES:
            if key not in data:
                # default assumptions
                if key in ["SeniorCitizen", "tenure"]:
                    data[key] = 0
                elif key == "Monthly_Charges":
                    data[key] = 0.0
                else:
                    data[key] = "No"

        return data
    except Exception as e:
        # fallback defaults
        return {
            key: 0 if key in ["SeniorCitizen", "tenure"] else 0.0 if key == "Monthly_Charges" else "No"
            for key in EXPECTED_FEATURES
        }
