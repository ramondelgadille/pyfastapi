# API Índice de Desequilibrio

Este proyecto implementa una API con FastAPI para predecir un índice de desequilibrio
(0 a 1) basado en cuatro variables:

- estado_n
- calidad_n
- deficit_n
- malestares_n

## Despliegue en Railway

1. Sube este repositorio a GitHub.
2. En Railway, crea un nuevo proyecto y selecciona "Deploy from GitHub Repo".
3. Railway detectará Python, instalará `requirements.txt` y usará el `Procfile`.
4. Cuando se despliegue, podrás acceder a `/predict` con una petición POST como esta:

```bash
curl -X POST https://<tu-app>.up.railway.app/predict   -H "Content-Type: application/json"   -d '{"features": [0.5, 0.3, 0.4, 0.2]}'
```

## Respuesta esperada

```json
{
  "vector": [0.5, 0.3, 0.4, 0.2],
  "salida": 0.57,
  "status": "riesgo"
}
```

## Ejecutar localmente

```bash
pip install -r requirements.txt
uvicorn predict_api:app --reload
```
