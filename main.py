from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en pruebas deja "*". Luego puedes poner solo el dominio de tu app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Input(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: Input):
    s = sum(data.features) if data.features else 1
    vector = [x / s for x in data.features] if s != 0 else data.features
    return {"status": "ok", "vector": vector}
