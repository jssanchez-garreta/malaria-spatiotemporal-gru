# ==============================================================================
# FEATURE ENGINEERING PIPELINE 
#
# 6 archivos de salida: X_train, X_val, X_test, Y_train, Y_val, Y_test
#
# Cada uno de ellos (con extensión .npy, tiene una salida del tipo (3740, 5, 5)
# El primer número es el total de muestras en ese archivo, el segundo es la 
# ventana (time steps) y el tercero el total de features (temperature, humidity,
# precipitation, precip_lag1 e incidence (histórico)
# ===============================================================================

import pandas as pd
import numpy as np

from google.colab import files
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. LOAD DATA
# ============================================================

uploaded = files.upload()
file_path = list(uploaded.keys())[0]

df = pd.read_csv(file_path)

print("Initial shape:", df.shape)

# ============================================================
# 2. SORT DATA
# ============================================================

df = df.sort_values(by=["country_region", "year"]).reset_index(drop=True)

# ============================================================
# 3. CREATE LAG FEATURES
# ============================================================

# Lag 1 precipitation (based on EDA)
df["precip_lag1"] = df.groupby("country_region")["precipitation"].shift(1)

# Remove rows with NaN from lag
df = df.dropna().reset_index(drop=True)

print("After lagging:", df.shape)

# ============================================================
# 4. NORMALIZATION (FIT ONLY ON TRAIN PERIOD)
# ============================================================

features = ["temperature", "precipitation", "humidity", "precip_lag1", "incidence"]

# Fit scaler ONLY on training years
train_mask_scaler = df["year"] <= 2016

scaler = StandardScaler()
scaler.fit(df.loc[train_mask_scaler, features])

# Apply scaling
df_scaled = df.copy()
df_scaled[features] = scaler.transform(df[features])

# ============================================================
# 5. CREATE SEQUENCES FROM FULL DATASET
# ============================================================

TIME_STEPS = 5

X, y, y_years = [], [], []

for region in df_scaled["country_region"].unique():

    region_df = df_scaled[df_scaled["country_region"] == region].sort_values("year")

    values = region_df[features].values
    years = region_df["year"].values

    for i in range(len(values) - TIME_STEPS):

        X.append(values[i:i+TIME_STEPS])
        y.append(values[i+TIME_STEPS, -1])  # incidence (last column)
        y_years.append(years[i+TIME_STEPS])

X = np.array(X)
y = np.array(y)
y_years = np.array(y_years)

# ============================================================
# 6. TEMPORAL SPLIT BASED ON TARGET YEAR
# ============================================================

train_mask = y_years <= 2016
val_mask = (y_years >= 2017) & (y_years <= 2020)
test_mask = y_years >= 2021

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

# ============================================================
# 7. FINAL CHECKS
# ============================================================

print("\nFinal dataset shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)

# ============================================================
# 8. SAVE ARRAYS (OPTIONAL)
# ============================================================

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("\n✅ Feature engineering completed successfully!")