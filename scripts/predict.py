import requests
import pandas as pd
from dataset import load_data, split_data


def predict(data: pd.DataFrame) -> list:
    """
    Llama a la API de MLflow con un DataFrame y devuelve las predicciones.
    """
    url = "http://127.0.0.1:6000/invocations"

    payload = {"dataframe_split": {
        "columns": data.columns.tolist(),
        "data":    data.values.tolist()
    }}

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["predictions"]


if __name__ == "__main__":
    # Cogemos el dataset de validación para probar
    X, y = load_data()
    _, X_val, _, _, y_val, _ = split_data(X, y)

    # Probamos con las primeras 5 muestras
    X_sample = X_val.head(5)
    y_real   = y_val.head(5).tolist()

    preds = predict(X_sample)

    print("\nComprobación de predicciones (primeras 5 muestras de validación):")
    print(f"{'Real':<10} {'Predicción':<10}")
    print("-" * 20)
    for real, pred in zip(y_real, preds):
        match = "✅" if real == pred else "❌"
        print(f"{real:<10} {pred:<10} {match}")