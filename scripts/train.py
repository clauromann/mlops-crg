import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset import load_data, split_data
from model import build_model


# ── Configuración MLflow ────────────────────────────────────────────────────
EXPERIMENT_NAME = "diabetes_experiment_prueba"
RUN_NAME        = "prueba_inicial"
MODEL_NAME      = "diabetes_model"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)


# ── Parámetros del modelo ───────────────────────────────────────────────────
PARAMS = {
    "C":        1.0,
    "max_iter": 1000,
    "solver":   "lbfgs"
}


# ── Entrenamiento ───────────────────────────────────────────────────────────
def run_training():
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    with mlflow.start_run(run_name=RUN_NAME):

        # 1. Registrar parámetros
        mlflow.log_params(PARAMS)

        # 2. Entrenar
        model = build_model(**PARAMS)
        model.fit(X_train, y_train)

        # 3. Evaluar sobre validación
        y_pred = model.predict(X_val)

        metrics = {
            "accuracy":  accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall":    recall_score(y_val, y_pred),
            "f1":        f1_score(y_val, y_pred),
        }

        # 4. Registrar métricas
        mlflow.log_metrics(metrics)

        # 5. Registrar el modelo
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        # 6. Imprimir resultados
        print(f"\nRun: {RUN_NAME}")
        print(f"Parámetros: {PARAMS}")
        print(f"Métricas (validación):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"\nModelo registrado como '{MODEL_NAME}' en el Registry.")


if __name__ == "__main__":
    run_training()