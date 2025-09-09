from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

