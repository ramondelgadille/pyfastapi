# main.py
# API mínima: entrena una Regresión Logística al arrancar y sirve /predict
# Entradas esperadas (todas normalizadas en [0,1], mayor=peor) en este orden:
# [estado_n, calidad_n, deficit_n, malestares_n]
# Respuesta: {"status":"ok","vector":[...],"salida_estres":prob}

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist
from typing import List
import numpy as np

# ====== IA: scikit-learn ======
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = FastAPI(title="Monstera Stress API (mínima)", version="0.1.0")

# CORS abierto para pruebas (ajusta en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ej.: ["https://app.flutterflow.io", "https://tu-dominio"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Esquema del request ----------
class Input(BaseModel):
    # EXACTAMENTE 4 features normalizadas [0,1]
    # [estado_n, calidad_n, deficit_n, malestares_n]
    features: conlist(float, min_items=4, max_items=4)

# ---------- Modelo global (entrenado al arrancar) ----------
pipe: Pipeline = None  # se inicializa en startup


# ---------- Generación de datos sintéticos para entrenar ----------
def _make_synthetic(n: int = 1500, seed: int = 7):
    rng = np.random.default_rng(seed)
    # Variables ya normalizadas y con mayor=peor, imitando tus discretizaciones típicas
    estado = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=n, p=[0.15, 0.25, 0.25, 0.2, 0.15])
    calidad = rng.choice([0.0, 0.5, 1.0], size=n, p=[0.4, 0.4, 0.2])
    deficit = rng.uniform(0, 1, size=n)                              # continuo [0,1]
    malestares = rng.choice([0.0, 1/3, 2/3, 1.0], size=n, p=[0.35, 0.3, 0.2, 0.15])

    X = np.vstack([estado, calidad, deficit, malestares]).T

    # Modelo verdadero oculto para generar etiquetas (probabilidad de "alto estrés")
    w_true = np.array([2.2, 1.3, 2.0, 1.6])
    b_true = -1.2
    z = X @ w_true + b_true
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(0, 1, size=n) < p).astype(int)  # clase binaria
    return X, y


# ---------- Entrenamiento mínimo ----------
def _train_model() -> Pipeline:
    X, y = _make_synthetic(n=2000, seed=9)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs"))
    ])
    pipe.fit(X, y)
    return pipe


@app.on_event("startup")
def _startup():
    global pipe
    pipe = _train_model()


# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"status": "ok", "model_loaded": pipe is not None, "version": app.version}

@app.post("/predict")
def predict(data: Input):
    global pipe
    if pipe is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")
    x = np.array(data.features, dtype=float).reshape(1, -1)

    # Validación simple de rango [0,1]
    if np.any(x < 0.0) or np.any(x > 1.0):
        raise HTTPException(status_code=400, detail="Todas las features deben estar en [0,1].")

    # Probabilidad de clase 1 (alto estrés)
    prob = float(pipe.predict_proba(x)[0, 1])

    return {
        "status": "ok",
        "vector": list(map(float, x.flatten())),
        "salida_estres": prob
    }
