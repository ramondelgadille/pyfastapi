from fastapi import FastAPI
from pydantic import BaseModel, conlist
from joblib import load
import numpy as np

MODEL_PATH = "model_desequilibrio.joblib"

model = load(MODEL_PATH)

app = FastAPI(
    title="API Índice de Desequilibrio",
    description="Predice el índice de desequilibrio (0-1) y devuelve un estado categórico.",
    version="1.0.0"
)

class FeaturesIn(BaseModel):
    features: conlist(float, min_items=4, max_items=4)
    # orden esperado: [estado_n, calidad_n, deficit_n, malestares_n]

def compute_status(valor):
    if valor < 0.4:
        return "estable"
    elif valor < 0.75:
        return "riesgo"
    return "alerta"

@app.post("/predict")
def predict(data: FeaturesIn):
    vector = np.array(data.features).reshape(1, -1)
    salida = float(model.predict(vector)[0])
    status = compute_status(salida)

    return {
        "vector": data.features,
        "salida": salida,
        "status": status
    }
