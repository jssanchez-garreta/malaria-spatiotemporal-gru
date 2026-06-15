# ============================================================
# RANDOM FOREST BASELINE (DESDE DATASET COMPLETO)
# ============================================================

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# 1. CARGAR DATASET
# ============================================================

df = pd.read_csv("dataset_with_lags.csv")

# ============================================================
# 2. SPLIT TEMPORAL
# ============================================================

train_df = df[df['year'] <= 2016]
test_df  = df[df['year'] >= 2021]

# ============================================================
# 3. FEATURES
# ============================================================

features = [
    'temperature',
    'precipitation',
    'humidity',
    'precip_lag1',
    'incidence_lag1'
]

# ============================================================
# 4. VARIABLES
# ============================================================

X_train = train_df[features]
y_train = train_df['incidence']

X_test = test_df[features]
y_test = test_df['incidence']

# ============================================================
# 5. MODELO
# ============================================================

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ============================================================
# 6. MÉTRICAS
# ============================================================

rf_preds = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

print("\n🌲 Random Forest Results")
print("MAE:", round(rf_mae, 3))
print("RMSE:", round(rf_rmse, 3))