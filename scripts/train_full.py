import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import build_model
from dataset import load_data, split_data

# ── Configuración MLflow ────────────────────────────────────────────────────
EXPERIMENT_NAME = "diabetes_experiment_completo"
MODEL_NAME      = "diabetes_model_final"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

# ── Grid de parámetros ──────────────────────────────────────────────────────
PARAM_GRID = {
    "C":         [0.01, 0.1, 1.0, 10.0],
    "solver":    ["lbfgs", "liblinear"],
    "max_iter":  [100, 500, 1000],
    "threshold": [0.5, 0.4, 0.3],
}


def evaluate(model, X, y):
    y_pred = model.predict(X)
    return {
        "accuracy":  round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred), 4),
        "recall":    round(recall_score(y, y_pred), 4),
        "f1":        round(f1_score(y, y_pred), 4),
    }


def run_full_training():
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    total = (len(PARAM_GRID["C"]) * len(PARAM_GRID["solver"]) *
             len(PARAM_GRID["max_iter"]) * len(PARAM_GRID["threshold"]))
    print(f"\nGrid Search sobre {total} combinaciones...\n")

    best_recall   = 0
    best_run_name = None

    for C in PARAM_GRID["C"]:
        for solver in PARAM_GRID["solver"]:
            for max_iter in PARAM_GRID["max_iter"]:
                for threshold in PARAM_GRID["threshold"]:

                    run_name = f"C={C}_solver={solver}_iter={max_iter}_thr={threshold}"

                    with mlflow.start_run(run_name=run_name):

                        model = build_model(C=C, solver=solver,
                                            max_iter=max_iter, threshold=threshold)
                        model.fit(X_train, y_train)
                        metrics = evaluate(model, X_val, y_val)

                        mlflow.log_params({
                            "C": C, "solver": solver,
                            "max_iter": max_iter, "threshold": threshold
                        })
                        mlflow.log_metrics(metrics)

                        if metrics["recall"] >= 0.70 and metrics["f1"] >= 0.65:
                            mlflow.sklearn.log_model(
                                sk_model=model,
                                artifact_path="model",
                                registered_model_name=MODEL_NAME,
                                code_paths=["scripts/model.py"]
                            )
                            print(f"  ✅ REGISTRADO  {run_name}")
                            print(f"     recall={metrics['recall']}  precision={metrics['precision']}  f1={metrics['f1']}")
                        else:
                            print(f"  ❌ descartado  {run_name} → recall={metrics['recall']} f1={metrics['f1']}")

                        if metrics["recall"] > best_recall:
                            best_recall   = metrics["recall"]
                            best_run_name = run_name

    print(f"\n{'─'*60}")
    print(f"Mejor recall overall: {best_recall:.4f} → {best_run_name}")
    print(f"Modelos registrados bajo '{MODEL_NAME}'")
    print(f"Abre http://127.0.0.1:5000 para comparar todos los runs.")


if __name__ == "__main__":
    run_full_training()