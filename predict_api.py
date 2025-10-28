# predict_api.py
from fastapi import FastAPI, HTTPException
from functools import lru_cache
import os
from joblib import load
import sklearn

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
app = FastAPI()

@lru_cache(maxsize=1)
def get_model():
    try:
        return load(MODEL_PATH)
    except Exception as e:
        # Log útil para debug
        print("SKLEARN VERSION:", sklearn.__version__)
        print("FAILED TO LOAD MODEL FROM:", MODEL_PATH)
        raise e

@app.get("/health")
def health():
    return {"sklearn": sklearn.__version__, "model_path": MODEL_PATH}

@app.post("/predict")
def predict(payload: dict):
    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model load error: {e.__class__.__name__}")
    # TODO: parsear features X del payload
    # y = model.predict([X])
    return {"ok": True}
