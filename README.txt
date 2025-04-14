 Problem Set 3 - Algorithmic Trading

En este trabajo, implementamos una estrategia de inversión automatizada basada en técnicas de Machine Learning aplicadas a series temporales financieras de los ETFs SPY (S&P 500) y QQQ (Nasdaq 100).

📚 Descripción del Proyecto
Este proyecto tiene como objetivo:

Construir y entrenar modelos de clasificación para predecir movimientos diarios (subida/bajada) en los ETFs SPY y QQQ.

Evaluar diferentes algoritmos como Decision Tree, Random Forest, Gradient Boosting, AdaBoost, LightGBM, XGBoost y CatBoost.

Simular una estrategia de inversión de $1000 en base a las predicciones generadas por los mejores modelos.

Maximizar el retorno esperado del portafolio con una asignación dinámica de capital basada en la confianza de los modelos.

⚙️ Estructura del Proyecto
css
Copiar
Editar
Pset3_AlgorithmicTrading/
│
├── data/
│   └── CLEAN/
│       ├── SPY_listo_entrenar_FINAL_READY.csv
│       └── QQQ_listo_entrenar_FINAL_READY.csv
│
├── models/
│   ├── GradientBoosting_SPY_REG.pkl
│   ├── GradientBoosting_QQQ_REG.pkl
│   ├── LightGBM_SPY.pkl
│   ├── LightGBM_QQQ.pkl
│   ├── CatBoost_SPY.pkl
│   ├── CatBoost_QQQ.pkl
│   ├── XGBoost_SPY.pkl
│   ├── XGBoost_QQQ.pkl
│   └── AdaBoost_Models.pkl
│
├── notebooks/
│   ├── Exploratory_Analysis.ipynb
│   ├── Modeling_LightGBM.ipynb
│   ├── Modeling_XGBoost.ipynb
│   ├── Modeling_CatBoost.ipynb
│   ├── Modeling_AdaBoost.ipynb
│   └── Portfolio_Simulation.ipynb
│
├── src/
│   └── modeling_utils.py
│
├── requirements.txt
├── README.md
└── Technical_Note.pdf
🛠️ Instalación y Configuración
Clonar el repositorio:

bash
Copiar
Editar
git clone https://github.com/tu_usuario/Pset3_AlgorithmicTrading.git
cd Pset3_AlgorithmicTrading
Crear un entorno virtual (opcional pero recomendado):

bash
Copiar
Editar
python -m venv venv
Activar el entorno virtual:

En Windows:

bash
Copiar
Editar
venv\Scripts\activate
En Mac/Linux:

bash
Copiar
Editar
source venv/bin/activate
Instalar todas las dependencias necesarias:

bash
Copiar
Editar
pip install -r requirements.txt
🏗️ Librerías Principales
pandas

numpy

matplotlib

scikit-learn

xgboost

lightgbm

catboost

imbalanced-learn

seaborn

joblib

scipy

Todas las librerías están listadas en requirements.txt.

📋 Metodología
Limpieza de Datos:
Eliminación de duplicados, manejo de valores nulos, normalización de variables, creación de retornos diarios y oversampling para balancear clases.

Feature Engineering:
Cálculo de indicadores técnicos (SMA, RSI, MACD, ATR, Bollinger Bands, etc.), generación de características de volumen, calendarios y lags temporales.

Modelado:
Entrenamiento y evaluación de varios modelos utilizando TimeSeriesSplit para evitar data leakage.
Optimización de hiperparámetros para minimizar overfitting.

Simulación de Portafolio:
Implementación de una simulación de inversión con reglas basadas en las predicciones de los modelos.

📈 Resultados
Capital inicial: $1000

Capital final después de la simulación: $1027.77

Modelos seleccionados:

Nasdaq (QQQ): Gradient Boosting

S&P 500 (SPY): XGBoost

Una evolución estable y positiva del capital fue lograda bajo un esquema de control de riesgo basado en predicciones diarias.

🧠 Lecciones Aprendidas
Importancia de respetar la estructura temporal para evitar data leakage.

Riesgos de overfitting en series financieras.

Ventajas de los modelos de boosting para capturar patrones complejos.

Valor de un buen \textit{feature engineering} y balance de clases en problemas financieros.

📄 Autores
Luciana Valdivieso

Nahomi

Mateo Salgado

Josué Cárdenas

Profesor: Erick Ñauñay

📜 Licencia
Este proyecto es únicamente para fines educativos (Universidad San Francisco de Quito - USFQ).

