from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import math

app = FastAPI(title="Monstera Stress API (baseline)", version="0.0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajusta en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Input(BaseModel):
    # features debe ser una lista de 4 floats en [0,1]
    features: list[float]

def sigmoid(z: float) -> float:
    z = max(min(z, 20.0), -20.0)
    return 1.0 / (1.0 + math.exp(-z))

@app.get("/")
def health():
    return {"status": "ok", "version": "0.0.2"}

@app.post("/predict")
def predict(data: Input):
    x = list(data.features)

    # 1) Longitud exacta
    if len(x) != 4:
        raise HTTPException(status_code=400, detail="Se esperan exactamente 4 features en el orden: [estado_n, calidad_n, deficit_n, malestares_n].")

    # 2) Rango [0,1]
    if any((v < 0.0 or v > 1.0) for v in x):
        raise HTTPException(status_code=400, detail="Todas las features deben estar en [0,1].")

    # Baseline lineal + sigmoide
    w = [2.2, 1.3, 2.0, 1.6]  # estado, calidad, deficit, malestares
    b = -1.2
    z = sum(wi * xi for wi, xi in zip(w, x)) + b
    salida_estres = sigmoid(z)

    return {"status": "ok", "vector": x, "salida_estres": salida_estres}
