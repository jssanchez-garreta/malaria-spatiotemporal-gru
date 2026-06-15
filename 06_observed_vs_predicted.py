# ============================================================
# REAL vs PREDICTED (WITH DENORMALIZATION)
# ============================================================

import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. DENORMALIZE
# ------------------------------------------------------------

mean_inc = scaler.mean_[-1]
std_inc = scaler.scale_[-1]

y_test_real = y_test * std_inc + mean_inc
y_pred_real = y_pred * std_inc + mean_inc

# Quick sanity check
print("\nSample predictions (real scale):")
for i in range(5):
    print(f"Real: {y_test_real[i]:.2f}, Pred: {y_pred_real[i]:.2f}")

# ------------------------------------------------------------
# 2. SCATTER PLOT
# ------------------------------------------------------------

plt.figure()

plt.scatter(y_test_real, y_pred_real, alpha=0.3)

# diagonal line
min_val = min(y_test_real.min(), y_pred_real.min())
max_val = max(y_test_real.max(), y_pred_real.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("Observed incidence")
plt.ylabel("Predicted incidence")
plt.title("Observed vs Predicted Malaria Incidence")

plt.savefig("observed_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.show()
