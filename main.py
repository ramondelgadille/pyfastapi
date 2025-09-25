from fastapi import FastAPI
from pydantic import BaseModel, conlist
from fastapi.middleware.cors import CORSMiddleware
import math

app = FastAPI(title="Monstera Stress API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción, restringe dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Input(BaseModel):
    # [estado_n, calidad_n, deficit_n, malestares_n] en [0,1]
    features: conlist(float, min_items=4, max_items=4)

def sigmoid(z: float) -> float:
    z = max(min(z, 20.0), -20.0)  # evita overflow
    return 1.0 / (1.0 + math.exp(-z))

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: Input):
    x = list(data.features)
    # baseline súper simple (si luego quieres cargar .joblib, se cambia aquí)
    w = [2.2, 1.3, 2.0, 1.6]
    b = -1.2
    z = sum(wi * xi for wi, xi in zip(w, x)) + b
    salida_estres = sigmoid(z)
    return {"status": "ok", "vector": x, "salida_estres": salida_estres}
