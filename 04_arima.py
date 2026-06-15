# ============================================================
# ARIMA BASELINE (DESDE DATASET CON LAGS)
# ============================================================

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm

# ============================================================
# 1. CARGAR DATASET
# ============================================================

df = pd.read_csv("dataset_with_lags.csv")

print("Shape dataset:", df.shape)

# ============================================================
# 2. SPLIT TEMPORAL (CONSISTENTE CON EL PAPER)
# ============================================================

train_df = df[df['year'] <= 2016]
test_df  = df[df['year'] >= 2021]

# ============================================================
# 3. INICIALIZAR CONTENEDORES
# ============================================================

arima_preds = []
arima_true  = []

regions = df['country_region'].unique()

# ============================================================
# 4. LOOP POR REGIÓN
# ============================================================

for region in regions:
    
    # --- Datos de la región ---
    train_series = train_df[train_df['country_region'] == region] \
                   .sort_values('year')['incidence']
    
    test_series = test_df[test_df['country_region'] == region] \
                  .sort_values('year')['incidence']
    
    # --- Seguridad ---
    if len(train_series) < 10 or len(test_series) == 0:
        continue
    
    try:
        # ====================================================
        # 5. ENTRENAR MODELO ARIMA
        # ====================================================
        
        model = pm.auto_arima(
            train_series,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_p=3,
            max_q=3,
            max_d=2
        )
        
        # ====================================================
        # 6. PREDICCIÓN
        # ====================================================
        
        preds = model.predict(n_periods=len(test_series))
        
        # ====================================================
        # 7. GUARDAR RESULTADOS
        # ====================================================
        
        arima_preds.extend(preds)
        arima_true.extend(test_series.values)
    
    except Exception as e:
        print(f"⚠️ Error en región {region}: {e}")
        continue

# ============================================================
# 8. MÉTRICAS GLOBALES
# ============================================================

arima_mae = mean_absolute_error(arima_true, arima_preds)
arima_rmse = np.sqrt(mean_squared_error(arima_true, arima_preds))

print("\n📈 ARIMA Results")
print("MAE:", round(arima_mae, 3))
print("RMSE:", round(arima_rmse, 3))

# ============================================================
# 9. SANITY CHECK
# ============================================================

print("\nPredicciones (primeros 5):")
print(arima_preds[:5])