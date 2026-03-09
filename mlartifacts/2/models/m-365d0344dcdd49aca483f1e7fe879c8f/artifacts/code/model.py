from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper que aplica un threshold personalizado sobre predict_proba.
    Así el threshold viaja dentro del modelo y se aplica automáticamente
    tanto en validación como cuando la API sirve predicciones.
    """
    def __init__(self, pipeline, threshold=0.5):
        self.pipeline  = pipeline
        self.threshold = threshold

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        proba = self.pipeline.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


def build_model(C=1.0, max_iter=1000, solver="lbfgs", threshold=0.5):
    pipeline = Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", LogisticRegression(C=C, max_iter=max_iter, solver=solver))
    ])
    return ThresholdClassifier(pipeline=pipeline, threshold=threshold)