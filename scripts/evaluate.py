import requests
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset import load_data, split_data

# Threshold usado en el modelo ganador
THRESHOLD = 0.4
API_URL   = "http://127.0.0.1:6000/invocations"


def predict_proba_api(data: pd.DataFrame) -> list:
    """
    Llama a la API y obtiene las probabilidades de clase 1 (diabetes).
    """
    payload = {"dataframe_split": {
        "columns": data.columns.tolist(),
        "data":    data.values.tolist()
    }}
    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    return response.json()["predictions"]


def evaluate_test():
    X, y = load_data()
    _, _, X_test, _, _, y_test = split_data(X, y)

    print(f"\nEvaluando sobre {len(X_test)} muestras de test...")

    # La API aplica el threshold internamente
    preds = predict_proba_api(X_test)
    y_pred = preds  # ya son clases 0/1 con threshold correcto

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
    }

    print(f"\n{'─'*40}")
    print(f"  MÉTRICAS REALES SOBRE TEST (10%)")
    print(f"{'─'*40}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v}")
    print(f"{'─'*40}")

    print(f"\n  Referencia validación (modelo ganador):")
    print(f"  recall=0.7963  precision=0.6935  f1=0.7414")
    print(f"\n  {'✅ Generaliza bien' if metrics['recall'] >= 0.75 else '⚠️  Posible overfitting — recall cae en test'}")

    return metrics


if __name__ == "__main__":
    evaluate_test()
