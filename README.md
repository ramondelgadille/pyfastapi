# API Índice de Desequilibrio

Este proyecto implementa una API con FastAPI para predecir un índice de desequilibrio
(0 a 1) basado en cuatro variables:

- estado_n
- calidad_n
- deficit_n
- malestares_n

## Respuesta esperada

```json
{
  "vector": [0.5, 0.3, 0.4, 0.2],
  "salida": 0.57,
  "status": "riesgo"
}
```
