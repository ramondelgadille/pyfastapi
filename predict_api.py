from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from joblib import load
import numpy as np
import os

MODEL_PATH = os.getenv("MODEL_PATH", "model_desequilibrio.joblib")

# Carga del modelo (con manejo de error para evitar crash en arranque)
try:
    model = load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"[startup] No se pudo cargar el modelo '{MODEL_PATH}': {e}")

app = FastAPI(
    title="API Índice de Desequilibrio",
    description="Predice el índice de desequilibrio (0-1) y devuelve un estado categórico.",
    version="1.0.0"
)

class FeaturesIn(BaseModel):
    # orden esperado: [estado_n, calidad_n, deficit_n, malestares_n]
    features: List[float] = Field(..., min_length=4, max_length=4)

def compute_status(valor):
    if valor < 0.4:
        return "estable"
    elif valor < 0.75:
        return "riesgo"
    return "alerta"

@app.get("/")
def health():
    return {
        "ok": True,
        "mensaje": "API viva",
        "modelo_cargado": model is not None
    }

@app.post("/predict")
def predict(data: FeaturesIn):
    if model is None:
        return {
            "error": "Modelo no cargado en el servidor. Verifica la ruta y versión de sklearn.",
            "modelo": MODEL_PATH
        }

    vector = np.array(data.features).reshape(1, -1)
    salida = float(model.predict(vector)[0])

    # si el modelo pudiera devolver fuera de [0,1], lo acotamos (opcional)
    salida = max(0.0, min(1.0, salida))

    status = compute_status(salida)
    return {
        "vector": data.features,
        "salida": salida,
        "status": status
    }
