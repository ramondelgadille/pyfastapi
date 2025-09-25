# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist, Field
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os, math
import numpy as np
import joblib

APP_TITLE = "Monstera Stress API"
APP_VERSION = "1.2.0"
MODEL_PATH = os.getenv("MODEL_PATH", "model/modelo_alto_estres.joblib")

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# Ajusta dominios en producción
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ej: ["https://app.flutterflow.io", "https://TU_DOMINIO"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Input(BaseModel):
    # [estado_n, calidad_n, deficit_n, malestares_n] en [0,1]
    features: conlist(float, min_items=4, max_items=4) = Field(..., description="Vector de 4 features normalizadas en [0,1]")

def sigmoid(z: float) -> float:
    z = max(min(z, 20.0), -20.0)
    return 1.0 / (1.0 + math.exp(-z))

# Cargar modelo si existe
pipe = None
if Path(MODEL_PATH).exists():
    try:
        pipe = joblib.load(MODEL_PATH)
        print(f"[INFO] Modelo cargado: {MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] No se pudo cargar el modelo ({e}). Se usará baseline.")

@app.get("/")
def root():
    return {"status": "ok", "version": APP_VERSION, "model_loaded": bool(pipe)}

@app.post("/predict")
def predict(data: Input):
    x = np.array(data.features, dtype=float).reshape(1, -1)
    if np.any(x < 0.0) or np.any(x > 1.0):
        raise HTTPException(status_code=400, detail="Todas las features deben estar en [0,1].")

    if pipe is not None:
        # Modelo real (clasificador) -> prob. de clase 1
        salida_estres = float(pipe.predict_proba(x)[0, 1])
    else:
        # Baseline (por si aún no entrenas)
        w = np.array([2.2, 1.3, 2.0, 1.6], dtype=float)  # estado, calidad, deficit, malestares
        b = -1.2
        z = float(x @ w.reshape(-1,1) + b)
        salida_estres = sigmoid(z)

    return {
        "status": "ok",
        "vector": list(map(float, x.flatten())),
        "salida_estres": salida_estres
    }
