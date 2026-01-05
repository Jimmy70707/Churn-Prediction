import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 1) Load and basic cleaning
df = pd.read_csv(DATA_PATH)
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# Ensure numeric types where appropriate
for col in ["tenure", "Monthly_Charges", "Total_Charges"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop features per your notebook (keep same names you used)
to_drop = [c for c in ["gender","Phone_Service","Dual","Streaming_TV","Streaming_Movies","Total_Charges"] if c in df.columns]
df = df.drop(columns=to_drop, errors="ignore")

# Target
if "Churn" not in df.columns:
    raise ValueError("Column 'Churn' not found. Make sure the CSV matches your notebook.")

# 2) Split features/target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})   # <-- FIXED
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 3) Identify columns by dtype
num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]

# 4) Preprocessing + model pipeline
pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

pipe = Pipeline(
    steps=[
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=5000, random_state=42))
    ]
)

# 5) Upsample for balance
df_bal = pd.concat([X, y], axis=1)
maj = df_bal[df_bal["Churn"] == 0]
minr = df_bal[df_bal["Churn"] == 1]
min_up = resample(minr, replace=True, n_samples=len(maj), random_state=42)
df_up = pd.concat([maj, min_up]).sample(frac=1, random_state=42).reset_index(drop=True)

Xb = df_up.drop(columns=["Churn"])
yb = df_up["Churn"].astype(int)

# 6) Train/test
X_train, X_test, y_train, y_test = train_test_split(Xb, yb, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)

# 7) Metrics (printed to console)
y_pred = pipe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))

# 8) Persist
joblib.dump(pipe, MODEL_DIR / "churn_pipeline.pkl")
meta = {
    "dropped_columns": to_drop,
    "numeric_columns": num_cols,
    "categorical_columns": cat_cols,
    "target": "Churn"
}
(MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
print("Saved models/churn_pipeline.pkl and models/meta.json")
