import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/diabetes.csv"

def load_data():
    """
    Carga el dataset y separa features (X) del target (y).
    - X: las 8 variables médicas
    - y: Outcome (0 = no diabetes, 1 = diabetes)
    """
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    return X, y


def split_data(X, y):
    """
    División: 70% train, 20% validación, 10% test.
    Usamos random_state=42 para reproducibilidad.
    """
    # Primero separamos el 10% de test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )
    # Del 90% restante, separamos validación (20% del total = ~22% de este trozo)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.222, random_state=42, stratify=y_temp
    )
    
    print(f"Train:      {len(X_train)} muestras ({len(X_train)/len(X)*100:.0f}%)")
    print(f"Validación: {len(X_val)} muestras ({len(X_val)/len(X)*100:.0f}%)")
    print(f"Test:       {len(X_test)} muestras ({len(X_test)/len(X)*100:.0f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X, y = load_data()
    split_data(X, y)