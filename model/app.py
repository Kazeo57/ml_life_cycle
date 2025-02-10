from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="API de prédiction avec FastAPI")

MODEL_PATH = "model/model.pkl"

# Chargement du modèle
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


model = load_model()

# Définition du format des données en entrée
class InputData(BaseModel):
    features: list[float]  


@app.post("/predict/")
async def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    try:
        X = np.array(data.features).reshape(1, -1) 
        prediction = model.predict(X).tolist() 
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API de prédiction 🚀"}