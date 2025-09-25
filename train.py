# train.py
import os
from pathlib import Path
import csv
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

DATA_PATH = os.getenv("DATA_PATH", "data/monstera_diario.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "model/modelo_alto_estres.joblib")

def load_from_csv(csv_path: str):
    X, y = [], []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = ["estado_n","calidad_n","deficit_n","malestares_n","salida_estres"]
        for row in reader:
            if not all(k in row for k in required):
                raise RuntimeError(f"CSV debe tener columnas: {required}")
            e = float(row["estado_n"])
            c = float(row["calidad_n"])
            d = float(row["deficit_n"])
            m = float(row["malestares_n"])
            s = float(row["salida_estres"])  # target continua 0..1
            # Clampeamos a [0,1] por seguridad
            x = [max(0,min(1,e)), max(0,min(1,c)), max(0,min(1,d)), max(0,min(1,m))]
            # Binarizamos etiqueta para clasificación
            y_bin = 1 if s >= 0.5 else 0
            X.append(x)
            y.append(y_bin)
    if not X:
        raise RuntimeError("CSV sin filas válidas.")
    return np.array(X, dtype=float), np.array(y, dtype=int)

def main():
    Path("model").mkdir(exist_ok=True)
    X, y = load_from_csv(DATA_PATH)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, solver="lbfgs"))
    ])
    pipe.fit(X_tr, y_tr)

    y_proba = pipe.predict_proba(X_te)[:,1]
    y_pred  = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_proba)
    print(f"Accuracy: {acc:.3f} | ROC-AUC: {auc:.3f}")
    print(classification_report(y_te, y_pred, digits=3))

    joblib.dump(pipe, MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")

if __name__ == "__main__":
    main()