import pandas as pd
import joblib
import numpy as np

# Charger le modèle et le préprocesseur
model = joblib.load("best_xgb_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Exemple d'un nouveau client (changer les valeurs selon le dataset)
new_client = pd.DataFrame({
    "gender": ["Male"],
    "SeniorCitizen": [0],
    "Partner": ["Yes"],
    "Dependents": ["No"],
    "tenure": [5],
    "PhoneService": ["Yes"],
    "MultipleLines": ["No"],
    "InternetService": ["Fiber optic"],
    "OnlineSecurity": ["No"],
    "OnlineBackup": ["Yes"],
    "DeviceProtection": ["No"],
    "TechSupport": ["No"],
    "StreamingTV": ["Yes"],
    "StreamingMovies": ["No"],
    "Contract": ["Month-to-month"],
    "PaperlessBilling": ["Yes"],
    "PaymentMethod": ["Electronic check"],
    "MonthlyCharges": [80.5],
    "TotalCharges": [400.0]
})

# Appliquer le prétraitement
new_client_transformed = preprocessor.transform(new_client)

# Prédiction
churn_probability = model.predict_proba(new_client_transformed)[:, 1][0]

# Affichage du résultat
print(f"Probabilité de churn pour ce client : {churn_probability:.2f}")
