# Churn Prediction Assistant
A machine learning system for predicting customer churn with natural language interface.

# Features
Dual Input Modes: Accepts both structured form data and natural language descriptions

Real-time Predictions: FastAPI backend with logistic regression model

Interactive Web Interface: Streamlit dashboard for easy interaction

Natural Language Processing: Extracts customer features from conversational text using Groq's LLM

Actionable Insights: Provides churn probability with business recommendations

# Project Structure

churn-prediction/

├── app/ # Backend API module

│   ├── train_model.py # Model training pipeline

│   ├── api.py             # FastAPI application

│   └── chatbot_llm.py     # LLM integration for NLP

│   └── streamlit_app.py       # Frontend web interface     

├── models/
│   └── churn_pipeline.pkl

│   └── log_reg_with_encoders.pkl

└── data/                 # Dataset folder
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv

    └── meta.json
    
├── requirements.txt       # Python dependencies

├── .env                  # Environment variables
