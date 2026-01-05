# 📊 Churn Prediction Assistant

An end-to-end **Customer Churn Prediction system** combining **Machine Learning**, **FastAPI**, **LLM-powered natural language understanding**, and a **Streamlit frontend**.

This project allows users to:
- Predict customer churn using structured data
- Ask questions in **plain natural language**
- Receive churn probability, predicted label, and actionable insights

---

## 🚀 Features

- **Logistic Regression churn model** trained on Telco Customer Churn data
- **FastAPI backend** for predictions and chat-based inference
- **LLM-powered chatbot** (Groq / LLaMA 3) to convert natural language into model-ready features
- **Streamlit web app** for interactive usage
- Automated **feature preprocessing and encoding**
- Human-readable **insights and recommendations**

---

## 🧠 System Architecture

churn-prediction/

├── app/ # Backend API module

│   ├── train_model.py # Model training pipeline

│   ├── api.py             # FastAPI application

│   └── chatbot_llm.py     # LLM integration for NLP

│   └── streamlit_app.py       # Frontend web interface     

├── models/

│   └── churn_pipeline.pkl

│   └── log_reg_with_encoders.pkl

├──data/

│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv

│   └── meta.json
    
├── requirements.txt       # Python dependencies

├── .env                  # Environment variables


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/jimmy70707/churn-prediction-assistant.git
cd churn-prediction-assistant
2️⃣ Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
3️⃣ Install dependencies

pip install -r requirements.txt
4️⃣ Set environment variables
Create a .env file:

GROQ_API_KEY=your_groq_api_key
API_URL=http://localhost:8080
```
### 🧪 Model Training (Optional)
To retrain the churn model:
```
python train_model.py
```
This will:

Balance the dataset

Train a Logistic Regression model

Save the pipeline to models/churn_pipeline.pkl

### 🖥️ Running the Application
Start the FastAPI backend
```
uvicorn api:app --host 0.0.0.0 --port 8080
```
API will be available at:

http://localhost:8080

http://localhost:8080/docs

Start the Streamlit frontend
```
streamlit run streamlit_app.py
```

App will open at:

http://localhost:8501

###  🔌 API Endpoints
GET /

Health check

{
  "message": "Churn Prediction API is running!"
}

POST /predict

Structured churn prediction

Request example:

{
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
}

Response : 

{
  "churn_label": "Yes",
  "churn_probability": 0.63,
  "insights": [
    "Month-to-month contracts are associated with higher churn risk.",
    "Customers with tenure below 12 months are more likely to churn."
  ]
  
}
POST /chat

Natural language churn analysis

Request example:


{
  "message": "A senior customer on month-to-month contract paying $90 per month"
}

Response :

{
  "churn_label": "Yes",
  "churn_probability": 0.71,
  "extracted_features": { ... },
  "insights": [ ... ]
}

## 🤖 LLM Integration
Uses Groq-hosted LLaMA 3

Converts free-text customer descriptions into structured JSON

Strict schema enforcement with fallback defaults

Zero-temperature for deterministic extraction

## 📊 Model Details
Algorithm: Logistic Regression

Preprocessing:

StandardScaler (numerical)

OneHotEncoder (categorical)

Class imbalance handled via upsampling

Output: churn probability + binary label

## 🛡️ Error Handling
Graceful fallback if LLM extraction fails

Input validation for missing or unseen categories

Safe defaults for incomplete user input

## 📌 Future Improvements
Add model explainability (SHAP)

Dockerize backend and frontend

Authentication & rate limiting

Model monitoring and drift detection

Cloud deployment (AWS / GCP / Azure)

## 📄 License
This project is licensed under the MIT License.

## 👤 Author
Muhammed Gamal
Machine Learning & AI Engineer
