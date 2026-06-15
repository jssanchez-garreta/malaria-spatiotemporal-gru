# ============================================================
# ERROR ANALYSIS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# INPUTS (must already be available in memory)
# ============================================================
# y_test_real, y_pred_real
# y_years, test_mask
# regions (aligned with sequences)

# ============================================================
# 1. BUILD MAIN DATAFRAME
# ============================================================

print("\nBuilding analysis dataframe...")

analysis_df = pd.DataFrame({
    "region": regions[test_mask],
    "year": y_years[test_mask],
    "real": y_test_real,
    "pred": y_pred_real
})

analysis_df["abs_error"] = np.abs(analysis_df["real"] - analysis_df["pred"])

# ============================================================
# 2. GLOBAL ERROR STATISTICS
# ============================================================

print("\n=== GLOBAL ERROR STATISTICS ===")
print(analysis_df["abs_error"].describe())

# ============================================================
# 3. ERROR BY REGION
# ============================================================

region_errors = (
    analysis_df
    .groupby("region")["abs_error"]
    .mean()
    .sort_values(ascending=False)
)

print("\n=== TOP 10 REGIONS (HIGHEST ERROR) ===")
print(region_errors.head(10))

print("\n=== TOP 10 REGIONS (LOWEST ERROR) ===")
print(region_errors.tail(10))

print("\n=== REGIONAL ERROR SUMMARY ===")
print(region_errors.describe())

# ============================================================
# 4. EXTREME VALUES ANALYSIS (TOP 10%)
# ============================================================

threshold_high = np.quantile(analysis_df["real"], 0.9)

print("\n=== EXTREME VALUES ANALYSIS ===")
print("High-incidence threshold (90th percentile):", round(threshold_high, 2))

high_df = analysis_df[analysis_df["real"] >= threshold_high]
normal_df = analysis_df[analysis_df["real"] < threshold_high]

print("\nHigh-incidence error stats:")
print(high_df["abs_error"].describe())

print("\nNormal cases error stats:")
print(normal_df["abs_error"].describe())

mean_high = high_df["abs_error"].mean()
mean_normal = normal_df["abs_error"].mean()

print("\nMean error (high-incidence):", round(mean_high, 3))
print("Mean error (normal cases):", round(mean_normal, 3))
print("Error ratio (high / normal):", round(mean_high / mean_normal, 3))

# ============================================================
# 5. TEMPORAL STABILITY ANALYSIS
# ============================================================

year_errors = (
    analysis_df
    .groupby("year")["abs_error"]
    .mean()
)

print("\n=== TEMPORAL ANALYSIS ===")
print("Error by year:")
print(year_errors)

print("\nSummary stats by year:")
print(year_errors.describe())

# ============================================================
# 6. PLOT: TEMPORAL ERROR
# ============================================================

plt.figure()
plt.plot(year_errors.index, year_errors.values, marker="o")

plt.xlabel("Year")
plt.ylabel("Mean Absolute Error")
plt.title("Temporal Stability of Prediction Error")

plt.savefig("temporal_error.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n✅ Error analysis completed successfully")