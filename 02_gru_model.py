# ============================================================
# GRU MODEL FOR MALARIA PREDICTION
# ============================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# 1. LOAD DATA
# ============================================================

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("Train shape:", X_train.shape)

# ============================================================
# 2. DEFINE MODEL
# ============================================================

model = Sequential([
    GRU(32, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse"
)

model.summary()

# ============================================================
# 3. EARLY STOPPING (IMPORTANT)
# ============================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ============================================================
# 4. TRAIN MODEL
# ============================================================

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ============================================================
# 5. EVALUATION ON TEST SET
# ============================================================

# Predictions
y_pred = model.predict(X_test).flatten()

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n✅ Model Performance on Test Set:")
print("MAE:", mae)
print("RMSE:", rmse)

# ============================================================
# 6. OPTIONAL: TRAINING HISTORY PLOT
# ============================================================

import matplotlib.pyplot as plt

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.title("Training and Validation Loss")
plt.savefig("training_loss.png", dpi=300, bbox_inches="tight")
plt.show()