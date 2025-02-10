from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd

app = FastAPI(title="API de prédiction avec FastAPI")

MODEL_PATH = "model/best_xgb_model.pkl"
PIPELINE_PATH = "model/preprocessor.pkl"

# Chargement du modèle
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def load_pipeline():
    if os.path.exists(PIPELINE_PATH):
        return joblib.load(PIPELINE_PATH)
    return None

model = load_model()
pipeline = load_pipeline()

# Définition du format des données en entrée
class InputData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict/")
async def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    try:
        input_df = pd.DataFrame([data.dict()])
        input_data = pipeline.transform(input_df)
        prediction = model.predict(input_data).tolist()
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API de prédiction 🚀"}