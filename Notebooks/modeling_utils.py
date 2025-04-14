# ===== modeling_utils.py =====

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split

# --------- Función para Entrenar y Evaluar un Modelo ---------

def train_and_evaluate_model(model, X, y, returns, test_size=0.2, n_splits=5):
    """
    Entrena, valida y evalúa un modelo usando validación de series de tiempo.

    Parámetros:
    - model: modelo sklearn o compatible.
    - X: features.
    - y: target.
    - returns: retornos diarios.
    - test_size: tamaño del test set.
    - n_splits: splits para TimeSeriesSplit.

    Retorna:
    - trained_model
    - metrics (dict)
    - X_train, X_test, y_train, y_test (para simular inversión después)
    """
    # Train/Test split respetando el tiempo
    X_train, X_test, y_train, y_test, returns_train, returns_test = train_test_split(
        X, y, returns, test_size=test_size, shuffle=False
    )

    model.fit(X_train, y_train)

    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluación
    metrics = {
        'Train Precision': precision_score(y_train, y_train_pred),
        'Test Precision': precision_score(y_test, y_test_pred),
        'Train Accuracy': accuracy_score(y_train, y_train_pred),
        'Test Accuracy': accuracy_score(y_test, y_test_pred),
        'Train ROC AUC': roc_auc_score(y_train, y_train_pred),
        'Test ROC AUC': roc_auc_score(y_test, y_test_pred),
        'Cumulative Return': simulate_returns(y_test, y_test_pred, returns_test)
    }
    
    return model, metrics, X_train, X_test, y_train, y_test, returns_test

# --------- Función para Guardar Modelos ---------

def save_model(model, filename):
    """Guarda el modelo entrenado como archivo .pkl."""
    joblib.dump(model, filename)
    print(f" Modelo guardado en {filename}")

# --------- Función para Cargar Modelos ---------

def load_model(filename):
    """Carga un modelo previamente guardado."""
    model = joblib.load(filename)
    print(f"Modelo cargado desde {filename}")
    return model

# --------- Función para Simular Ganancia ---------

def simulate_returns(y_true, y_pred, returns):
    """
    Simula ganancia acumulada basada en predicciones.
    Compra si predices subida (1), vende o no opera si predices bajada (0).
    """
    strat_returns = returns * (2 * y_pred - 1)  # +r si predice 1, -r si predice 0
    cumulative_returns = (strat_returns + 1).cumprod() - 1
    return cumulative_returns.iloc[-1]  # retorno acumulado final

# --------- Función para Graficar Ganancia ---------

def plot_cumulative_returns(y_true, y_pred, returns, title="Cumulative Returns"):
    """
    Grafica el crecimiento del portafolio según las predicciones.
    """
    strat_returns = returns * (2 * y_pred - 1)
    cumulative_returns = (strat_returns + 1).cumprod()

    plt.figure(figsize=(10,6))
    plt.plot(cumulative_returns, label="Strategy")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid()
    plt.show()
