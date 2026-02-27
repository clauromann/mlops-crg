"""
app-iris-ct: Continuous Training extension for ML-FastAPI-Docker
================================================================
Extends app-iris with:
  - POST /train      → reentrenamiento incremental con nuevas muestras
  - GET  /model/info → versión activa, métricas, historial
  - POST /predict    → inferencia (igual que app-iris, con versión activa)
  - GET  /health     → estado del servicio

EXTENSIÓN (Ejercicio de programación):
  - /train acepta un campo policy con 3 modos:
      * any_improvement (>= actual)
      * min_delta (mejora mínima configurable, ej. 2%)
      * per_class_f1 (activar solo si mejora el F1 de una clase específica)
  - El historial registra la policy usada y el motivo de activación/rechazo.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "model_active.joblib"
HISTORY_PATH = MODELS_DIR / "training_history.json"
DATA_PATH = MODELS_DIR / "accumulated_data.joblib"

# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Iris Continuous Training API",
    version="1.0.0",
    description="Extensión MLOps de app-iris. Sirve predicciones y permite reentrenar "
                "el modelo con nuevas muestras etiquetadas, registrando el historial de versiones."
)

# ---------------------------------------------------------------------------
# Esquemas Pydantic
# ---------------------------------------------------------------------------

class IrisSample(BaseModel):
    sepal_length: float = Field(..., example=5.1, description="Longitud del sépalo (cm)")
    sepal_width: float  = Field(..., example=3.5, description="Anchura del sépalo (cm)")
    petal_length: float = Field(..., example=1.4, description="Longitud del pétalo (cm)")
    petal_width: float  = Field(..., example=0.2, description="Anchura del pétalo (cm)")


class LabeledSample(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float  = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float  = Field(..., example=0.2)
    label: int = Field(..., ge=0, le=2, example=0,
                       description="0=setosa, 1=versicolor, 2=virginica")


PolicyMode = Literal["any_improvement", "min_delta", "per_class_f1"]


class TrainRequest(BaseModel):
    samples: List[LabeledSample] = Field(
        ..., min_items=5,
        description="Nuevas muestras etiquetadas para reentrenamiento (mínimo 5)"
    )
    retrain_from_scratch: bool = Field(
        False,
        description="Si True, ignora datos anteriores y entrena solo con las muestras enviadas"
    )

    # NUEVO: política de activación
    policy: PolicyMode = Field(
        "any_improvement",
        description="Política de activación del nuevo modelo: "
                    "any_improvement | min_delta | per_class_f1"
    )

    # NUEVO: parámetros (solo aplican a ciertas policies)
    min_delta: Optional[float] = Field(
        None,
        description="Solo para policy='min_delta'. Mejora mínima exigida (ej 0.02 = 2%)",
        example=0.02
    )
    target_class: Optional[int] = Field(
        None,
        description="Solo para policy='per_class_f1'. Clase objetivo (0,1,2) cuyo F1 debe mejorar",
        example=1,
        ge=0, le=2
    )


class PredictResponse(BaseModel):
    prediction: int
    class_name: str
    model_version: str


class TrainResponse(BaseModel):
    status: str
    model_version: str
    accuracy_new: float
    accuracy_previous: Optional[float]
    model_updated: bool
    message: str


class ModelInfo(BaseModel):
    active_version: str
    trained_at: str
    accuracy: float
    n_training_samples: int
    algorithm: str
    history: List[dict]


# ---------------------------------------------------------------------------
# Utilidades de persistencia / constantes
# ---------------------------------------------------------------------------
CLASS_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}
ALL_LABELS = [0, 1, 2]


def load_history() -> List[dict]:
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return []


def save_history(history: List[dict]):
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def get_active_model_meta() -> Optional[dict]:
    history = load_history()
    return history[-1] if history else None


# ---------------------------------------------------------------------------
# ActivationPolicy (NUEVO)
# ---------------------------------------------------------------------------
class ActivationPolicy:
    """
    Decide si se activa el nuevo modelo según una policy configurable.
    Devuelve: (activate: bool, reason: str)
    """

    @staticmethod
    def decide(
        policy: PolicyMode,
        accuracy_new: float,
        accuracy_prev: Optional[float],
        f1_new: Dict[int, float],
        f1_prev: Dict[int, float],
        min_delta: Optional[float] = None,
        target_class: Optional[int] = None
    ) -> Tuple[bool, str]:

        # Caso inicial: si no hay modelo anterior, activamos siempre
        if accuracy_prev is None:
            return True, "Primer modelo: no existe modelo anterior."

        if policy == "any_improvement":
            if accuracy_new >= accuracy_prev:
                return True, f"Activado (any_improvement): {accuracy_new:.4f} >= {accuracy_prev:.4f}"
            return False, f"Rechazado (any_improvement): {accuracy_new:.4f} < {accuracy_prev:.4f}"

        if policy == "min_delta":
            # Si no viene, usamos 2% por defecto (ejemplo del enunciado)
            delta = 0.02 if min_delta is None else float(min_delta)
            if accuracy_new >= (accuracy_prev + delta):
                return True, f"Activado (min_delta): {accuracy_new:.4f} >= {accuracy_prev:.4f} + {delta:.4f}"
            return False, f"Rechazado (min_delta): {accuracy_new:.4f} < {accuracy_prev:.4f} + {delta:.4f}"

        if policy == "per_class_f1":
            if target_class is None:
                return False, "Rechazado (per_class_f1): falta target_class."
            tc = int(target_class)
            new_tc = f1_new.get(tc, 0.0)
            prev_tc = f1_prev.get(tc, 0.0)
            if new_tc > prev_tc:
                return True, f"Activado (per_class_f1): F1 clase {tc} mejora {prev_tc:.4f} → {new_tc:.4f}"
            return False, f"Rechazado (per_class_f1): F1 clase {tc} no mejora ({prev_tc:.4f} → {new_tc:.4f})"

        return False, "Rechazado: policy desconocida."


# ---------------------------------------------------------------------------
# Bootstrap: si no existe modelo, lo entrenamos con el dataset original
# ---------------------------------------------------------------------------
def bootstrap_model():
    """Entrena un modelo base con el dataset Iris completo al arrancar."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1_arr = f1_score(y_test, y_pred, labels=ALL_LABELS, average=None, zero_division=0)
    f1_per_class = {int(lbl): float(f1_arr[i]) for i, lbl in enumerate(ALL_LABELS)}

    version = "v1.0-base"
    joblib.dump(clf, MODEL_PATH)

    history = [{
        "version": version,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "accuracy": round(accuracy, 4),
        "f1_per_class": {str(k): round(v, 4) for k, v in f1_per_class.items()},
        "n_training_samples": len(X_train),
        "algorithm": "LogisticRegression",
        "source": "bootstrap (iris dataset completo)",
        "activated": True,
        "status": "activado",
        "policy": "any_improvement",
        "policy_params": {},
        "decision_reason": "Bootstrap inicial"
    }]

    save_history(history)
    print(f"[bootstrap] Modelo base creado → versión={version}, accuracy={accuracy:.4f}")


@app.on_event("startup")
def startup_event():
    if not MODEL_PATH.exists():
        bootstrap_model()
    else:
        meta = get_active_model_meta()
        if meta:
            print(f"[startup] Modelo activo cargado → versión={meta.get('version','unknown')}, "
                  f"accuracy={meta.get('accuracy','?')}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Sistema"])
def health():
    meta = get_active_model_meta()
    return {
        "status": "ok",
        "active_model_version": meta["version"] if meta else "none",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inferencia"])
def predict(sample: IrisSample):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Modelo no disponible. Llama primero a /train.")

    clf = joblib.load(MODEL_PATH)

    X = np.array([[
        sample.sepal_length,
        sample.sepal_width,
        sample.petal_length,
        sample.petal_width
    ]])

    pred = int(clf.predict(X)[0])
    meta = get_active_model_meta()

    return PredictResponse(
        prediction=pred,
        class_name=CLASS_NAMES[pred],
        model_version=meta["version"] if meta else "unknown"
    )


@app.post("/train", response_model=TrainResponse, tags=["Entrenamiento"])
def train(request: TrainRequest):
    """
    Reentrena el modelo con las nuevas muestras enviadas.

    - retrain_from_scratch=False: acumula muestras y reentrena sobre el total.
    - retrain_from_scratch=True: entrena solo con las muestras enviadas.
    - NUEVO: policy configurable para activar / rechazar.
    - Siempre registra el intento en el historial (activado o rechazado).
    """

    # 1) Preparar nuevas muestras
    new_X = np.array([[s.sepal_length, s.sepal_width, s.petal_length, s.petal_width]
                      for s in request.samples])
    new_y = np.array([s.label for s in request.samples])

    # 2) Recuperar info del modelo activo actual
    history = load_history()
    previous_accuracy = history[-1]["accuracy"] if history else None

    # F1 del modelo anterior (si no existe en history, ponemos 0.0)
    previous_f1 = {0: 0.0, 1: 0.0, 2: 0.0}
    if history and "f1_per_class" in history[-1]:
        prev_dict = history[-1]["f1_per_class"]
        # puede venir como strings si ya estaba guardado así
        previous_f1 = {int(k): float(v) for k, v in prev_dict.items()}

    # 3) Construir dataset de entrenamiento
    if (not request.retrain_from_scratch) and DATA_PATH.exists():
        saved = joblib.load(DATA_PATH)
        X_train = np.vstack([saved["X"], new_X])
        y_train = np.concatenate([saved["y"], new_y])
        source = f"incremental (+{len(new_X)} muestras nuevas, {len(saved['X'])} anteriores)"
    else:
        X_train, y_train = new_X, new_y
        source = f"desde cero ({len(new_X)} muestras)"

    # 4) Reglas mínimas del LAB
    if len(np.unique(y_train)) < 2:
        raise HTTPException(
            status_code=422,
            detail="El dataset de entrenamiento debe contener al menos 2 clases distintas."
        )

    # 5) Entrenar el modelo
    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X_train, y_train)

    # 6) Evaluación
    #    (mantenemos la lógica del LAB: con <20 evaluamos en train)
    if len(y_train) < 20:
        y_eval = y_train
        X_eval = X_train
        eval_note = "evaluación en train (dataset pequeño, < 20 muestras)"
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        clf = LogisticRegression(max_iter=200, random_state=42)
        clf.fit(X_tr, y_tr)
        X_eval, y_eval = X_te, y_te
        eval_note = f"validación con {len(y_te)} muestras"

    y_pred = clf.predict(X_eval)
    accuracy_new = float(accuracy_score(y_eval, y_pred))

    f1_arr = f1_score(y_eval, y_pred, labels=ALL_LABELS, average=None, zero_division=0)
    f1_new = {int(lbl): float(f1_arr[i]) for i, lbl in enumerate(ALL_LABELS)}

    # 7) Decisión según policy
    activate, reason = ActivationPolicy.decide(
        policy=request.policy,
        accuracy_new=accuracy_new,
        accuracy_prev=previous_accuracy,
        f1_new=f1_new,
        f1_prev=previous_f1,
        min_delta=request.min_delta,
        target_class=request.target_class
    )

    # 8) Versionado
    version = f"v{len(history) + 1}.0-{uuid.uuid4().hex[:6]}"

    # 9) Guardar intento en historial (siempre)
    entry = {
        "version": version,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "accuracy": round(accuracy_new, 4),
        "f1_per_class": {str(k): round(v, 4) for k, v in f1_new.items()},
        "n_training_samples": int(len(X_train)),
        "algorithm": "LogisticRegression",
        "source": source,
        "eval_note": eval_note,
        "policy": request.policy,
        "policy_params": {
            **({"min_delta": request.min_delta} if request.min_delta is not None else {}),
            **({"target_class": request.target_class} if request.target_class is not None else {}),
        },
        "decision_reason": reason,
        "activated": bool(activate),
        "status": "activado" if activate else "rechazado"
    }

    history.append(entry)
    save_history(history)

    # 10) Si activamos, persistimos el modelo activo y el dataset acumulado
    if activate:
        joblib.dump(clf, MODEL_PATH)
        joblib.dump({"X": X_train, "y": y_train}, DATA_PATH)

        message = f"Nuevo modelo activado. {reason}"
        return TrainResponse(
            status="activado",
            model_version=version,
            accuracy_new=round(accuracy_new, 4),
            accuracy_previous=previous_accuracy,
            model_updated=True,
            message=message
        )

    message = f"Modelo NO activado. {reason}"
    return TrainResponse(
        status="rechazado",
        model_version=version,
        accuracy_new=round(accuracy_new, 4),
        accuracy_previous=previous_accuracy,
        model_updated=False,
        message=message
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Modelo"])
def model_info():
    history = load_history()
    if not history:
        raise HTTPException(status_code=404, detail="No hay historial de modelos.")

    active = history[-1]
    return ModelInfo(
        active_version=active["version"],
        trained_at=active["trained_at"],
        accuracy=active["accuracy"],
        n_training_samples=active["n_training_samples"],
        algorithm=active.get("algorithm", "unknown"),
        history=history
    )


@app.delete("/model/history", tags=["Modelo"])
def reset_history():
    """
    Resetea historial y elimina ficheros de modelos para pruebas.
    """
    # Borra artefactos si existen
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
    if DATA_PATH.exists():
        DATA_PATH.unlink()

    # Rebootstrapping
    bootstrap_model()
    return {"status": "ok", "message": "Historial eliminado. Modelo base restaurado."}