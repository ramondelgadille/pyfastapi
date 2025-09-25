from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from fastapi.middleware.cors import CORSMiddleware
import math

app = FastAPI(title="Monstera Stress API (baseline)", version="0.0.1")

# CORS abierto para pruebas; ajusta dominios en producción
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Input(BaseModel):
    # EXACTAMENTE 4 features normalizadas [0..1] en este orden:
    # [estado_n, calidad_n, deficit_n, malestares_n]
    features: conlist(float, min_items=4, max_items=4)

def sigmoid(z: float) -> float:
    # recorte para evitar overflow en exp
    z = max(min(z, 20.0), -20.0)
    return 1.0 / (1.0 + math.exp(-z))

@app.get("/")
def health():
    return {"status": "ok", "version": "0.0.1"}

@app.post("/predict")
def predict(data: Input):
    x = list(data.features)  # [estado_n, calidad_n, deficit_n, malestares_n]
    # validación de rango
    if any((v < 0.0 or v > 1.0) for v in x):
        raise HTTPException(status_code=400, detail="Todas las features deben estar en [0,1].")
    # baseline lineal + sigmoide (pesos fijos razonables)
    w = [2.2, 1.3, 2.0, 1.6]  # estado, calidad, deficit, malestares
    b = -1.2
    z = sum(wi * xi for wi, xi in zip(w, x)) + b
    salida_estres = sigmoid(z)
    return {"status": "ok", "vector": x, "salida_estres": salida_estres}
