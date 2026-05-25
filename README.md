# pyfastapi — Imbalance Index Prediction API for Monstera

A REST API built with **FastAPI** and deployed on **Railway**, exposing a trained **Random Forest** model that predicts a person's imbalance index based on four weighted wellness indicators. This is the machine learning backend that the **Monstera** app connects to via API.

---

## How it works

```
Monstera App  →  HTTP POST request  →  FastAPI on Railway  →  loads model_desequilibrio.joblib  →  prediction  →  JSON response
```

The `model_desequilibrio.joblib` file contains the Random Forest model **already trained and serialized**. Training happened once, and the model was saved as a file. When the API receives a request, it simply loads it and returns a prediction instantly — no retraining needed. This ensures fast responses and consistent results.

---

## Input variables

The model receives exactly **4 normalized float values** (between 0 and 1), in this order:

| # | Variable | Description |
|---|----------|-------------|
| 1 | `estado_n` | Perceived general state |
| 2 | `calidad_n` | Quality of life or sleep |
| 3 | `deficit_n` | Perceived deficit or lack |
| 4 | `malestares_n` | Level of physical or emotional discomfort |

These variables correspond to the **weighted stress indicators** studied in the Monstera theoretical model.

---

## Model output

The API returns the imbalance index (`salida`) in a 0–1 range, along with a categorical status:

| Range | Status |
|-------|--------|
| `< 0.4` | ✅ `estable` (stable) |
| `0.4 – 0.74` | ⚠️ `riesgo` (at risk) |
| `>= 0.75` | 🔴 `alerta` (alert) |

> Status labels are kept in Spanish as they are consumed directly by the Monstera app.

---

## Endpoints

### `GET /`
Health check. Verifies the API is running and the model is loaded.

**Example response:**
```json
{
  "ok": true,
  "mensaje": "API viva",
  "modelo_cargado": true
}
```

---

### `POST /predict`
Receives the 4 indicators and returns the prediction.

**Request body (JSON):**
```json
{
  "features": [0.6, 0.4, 0.7, 0.8]
}
```

**Example response:**
```json
{
  "vector": [0.6, 0.4, 0.7, 0.8],
  "salida": 0.82,
  "status": "alerta"
}
```

---

## Project structure

```
pyfastapi-monstera/
├── predict_api.py               # Main API code
├── model_desequilibrio.joblib   # Trained Random Forest model (serialized)
└── requirements.txt             # Project dependencies
```

---

## Dependencies

```
fastapi
uvicorn
pydantic
joblib
numpy
scikit-learn
```

---

## Railway deployment

1. Connect this repository on [railway.app](https://railway.app)
2. Railway automatically detects the Python environment
3. Make sure the start command is:
   ```
   uvicorn predict_api:app --host 0.0.0.0 --port $PORT
   ```
4. The `MODEL_PATH` environment variable is optional — it defaults to `model_desequilibrio.joblib` at the project root

---

## Connecting from Monstera

The Monstera app should send a `POST /predict` request to the public URL assigned by Railway, with the 4 normalized indicators as an array in `features`.

Example using `fetch` (JavaScript):
```js
const response = await fetch("https://your-api.up.railway.app/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ features: [0.6, 0.4, 0.7, 0.8] })
});
const data = await response.json();
console.log(data.status); // "estable" | "riesgo" | "alerta"
```
