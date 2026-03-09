# MLOps — Diabetes MLflow Project

Proyecto MLOps para clasificación binaria de diabetes usando MLflow Tracking, MLflow Model Registry y scikit-learn.

- **Dataset:** Pima Indians Diabetes — 768 muestras, 8 features médicas, target binario (0 = no diabetes, 1 = diabetes).
- **Modelo:** Pipeline de StandardScaler + LogisticRegression con threshold de decisión personalizado.
- **Objetivo:** Maximizar recall (detectar el máximo de casos reales de diabetes), con un umbral mínimo de recall ≥ 0.70 y f1 ≥ 0.65.

---

## Estructura del proyecto

```
diabetes-mlflow/
├── data/
│   └── diabetes.csv          ← dataset original
├── scripts/
│   ├── dataset.py            ← carga y split (70% train / 20% val / 10% test)
│   ├── model.py              ← ThresholdClassifier: pipeline + threshold embebido
│   ├── train.py              ← Parte I: experimento de prueba (modelo base)
│   ├── train_full.py         ← Parte II: grid search completo (72 combinaciones)
│   ├── predict.py            ← comprobación rápida de predicciones via API
│   └── evaluate.py           ← evaluación final sobre test set via API
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

> Todos los comandos siguientes asumen que el venv está activado y que estás en el directorio raíz del proyecto (`diabetes-mlflow/`).

---

## Parte I — Experimento de prueba

Entrena un modelo base con parámetros fijos y lo registra en el MLflow Model Registry. Sirve para verificar que el stack funciona de principio a fin antes de lanzar el grid search.

Requiere **3 terminales** con el venv activado.

### Terminal 1 — Servidor MLflow (tracking + registry)

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Mantener activo durante toda la sesión. Accede a la UI en: http://127.0.0.1:5000

### Terminal 2 — Entrenar y registrar el modelo base

```bash
python scripts/train.py
```

Esto entrena con `C=1.0, solver=lbfgs, threshold=0.5`, evalúa sobre validación y registra el modelo en el Registry bajo el nombre `diabetes_model`.

### Terminal 3 — Servir el modelo via API

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
mlflow models serve -m "models:/diabetes_model/1" --host 127.0.0.1 --port 6000 --no-conda
```

### Terminal 2 — Comprobar predicciones

```bash
python scripts/predict.py
```

Envía 5 muestras del set de validación a la API y muestra las predicciones frente a los valores reales.

---

## Parte II — Experimentos completos con Grid Search

Grid search sobre 72 combinaciones de hiperparámetros con criterio médico de selección. El threshold de decisión se embebe directamente en el modelo para que la API lo aplique automáticamente.

**Parámetros explorados:**

| Parámetro   | Valores                    |
|-------------|----------------------------|
| `C`         | 0.01, 0.1, 1.0, 10.0       |
| `solver`    | lbfgs, liblinear           |
| `max_iter`  | 100, 500, 1000             |
| `threshold` | 0.5, 0.4, 0.3              |

**Criterio de registro:** solo se registran en el Model Registry los modelos que cumplen `recall ≥ 0.70` **y** `f1 ≥ 0.65` sobre el set de validación.

### Paso 1 — Lanzar el grid search (Terminal 2)

> La Terminal 1 debe seguir con el servidor MLflow activo.

```bash
python scripts/train_full.py
```

El script imprime cada combinación indicando si fue registrada (✅) o descartada (❌). Al finalizar muestra la combinación con mejor recall global.

Consulta todos los runs en: http://127.0.0.1:5000 → Experiment `diabetes_experiment_completo`

### Paso 2 — Seleccionar y promover el modelo ganador

1. Abre http://127.0.0.1:5000 → **Models** → `diabetes_model_final`
2. Busca el modelo con mejor equilibrio entre recall y f1 (criterio médico: priorizar recall sin sacrificar demasiado precisión)
3. Haz clic en esa versión → **Stage** → **Production**
   - Marca la casilla para archivar la versión anterior en Production
   - Añade el comentario: `Mejor modelo`

### Paso 3 — Servir el modelo de Production (Terminal 2)

Para la API anterior si sigue activa (Ctrl+C) y lanza:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
mlflow models serve -m "models:/diabetes_model_final/Production" --host 127.0.0.1 --port 6000 --no-conda
```

Fíjate en que se carga por stage (`Production`) en vez de por número de versión — así siempre sirves el modelo activo sin hardcodear versiones.

### Paso 4 — Evaluar sobre el test set (Terminal 3)

```bash
python scripts/evaluate.py
```

Evalúa el modelo de Production sobre las **77 muestras de test** (10% del dataset, no vistas durante el entrenamiento ni la selección). El threshold se aplica automáticamente dentro del modelo.


## Notas técnicas

- **ThresholdClassifier:** el threshold personalizado viaja embebido dentro del objeto modelo gracias al wrapper `ThresholdClassifier` definido en `model.py`. Esto garantiza que la API de MLflow aplica el mismo threshold que se usó durante la evaluación en validación.
- **Criterio médico:** se prioriza recall sobre precisión porque un falso negativo (no detectar un caso real de diabetes) es clínicamente más peligroso que un falso positivo.
- **max_iter:** no tiene efecto significativo en este dataset — el modelo converge en pocas iteraciones independientemente del valor.
- **C alto vs bajo:** a partir de C=0.1 las métricas se estabilizan. C=0.01 con liblinear y threshold=0.4 da recall alto pero a costa de peor precisión.


