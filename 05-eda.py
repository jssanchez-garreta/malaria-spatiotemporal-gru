# ============================================================
# STEP: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional styling (clean for paper)
sns.set(style="whitegrid")

# ============================================================
# 1. LOAD DATA  (final_dataset.csv)
# ============================================================

from google.colab import files
uploaded = files.upload()

file_path = list(uploaded.keys())[0]
df = pd.read_csv(file_path)

print("Dataset loaded:", df.shape)

# ============================================================
# 2. DISTRIBUTION OF INCIDENCE
# ============================================================

plt.figure()
plt.hist(df["incidence"], bins=50)
plt.title("Distribution of Malaria Incidence")
plt.xlabel("Incidence (cases per 1,000 population)")
plt.ylabel("Frequency")
plt.savefig("incidence_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 3. GLOBAL TEMPORAL TREND
# ============================================================

annual_incidence = df.groupby("year")["incidence"].mean()

plt.figure()
plt.plot(annual_incidence.index, annual_incidence.values)
plt.title("Global Trend of Malaria Incidence (2000–2024)")
plt.xlabel("Year")
plt.ylabel("Mean Incidence")
plt.savefig("incidence_trend.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 4. SPATIAL VARIABILITY (BOXPLOT)
# ============================================================

# Reduce number of regions shown (top variability)
sample_regions = df.groupby("country_region")["incidence"].mean().sort_values().index[:20]

subset = df[df["country_region"].isin(sample_regions)]

plt.figure(figsize=(10, 5))
sns.boxplot(data=subset, x="country_region", y="incidence")
plt.xticks(rotation=90)
plt.title("Spatial Variability of Incidence (Selected Regions)")
plt.xlabel("Region")
plt.ylabel("Incidence")
plt.savefig("spatial_variability.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 5. CLIMATE vs INCIDENCE (SCATTER PLOTS)
# ============================================================

# Temperature
plt.figure()
plt.scatter(df["temperature"], df["incidence"], alpha=0.3)
plt.title("Incidence vs Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("Incidence")
plt.savefig("incidence_vs_temperature.png", dpi=300, bbox_inches="tight")
plt.close()

# Precipitation
plt.figure()
plt.scatter(df["precipitation"], df["incidence"], alpha=0.3)
plt.title("Incidence vs Precipitation")
plt.xlabel("Precipitation (mm)")
plt.ylabel("Incidence")
plt.savefig("incidence_vs_precipitation.png", dpi=300, bbox_inches="tight")
plt.close()

# Humidity
plt.figure()
plt.scatter(df["humidity"], df["incidence"], alpha=0.3)
plt.title("Incidence vs Humidity")
plt.xlabel("Relative Humidity (%)")
plt.ylabel("Incidence")
plt.savefig("incidence_vs_humidity.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 6. CORRELATION MATRIX
# ============================================================

corr = df[[
    "incidence",
    "temperature",
    "precipitation",
    "humidity"
]].corr()

plt.figure()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig("correlation_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 7. LAG ANALYSIS (PRECIPITATION)
# ============================================================

df = df.sort_values(["country_region", "year"])
df["precip_lag1"] = df.groupby("country_region")["precipitation"].shift(1)

lag_df = df.dropna()

plt.figure()
plt.scatter(lag_df["precip_lag1"], lag_df["incidence"], alpha=0.3)
plt.title("Incidence vs Precipitation (Lag 1 Year)")
plt.xlabel("Precipitation (lag 1 year)")
plt.ylabel("Incidence")
plt.savefig("lag_analysis_precipitation.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 8. SAMPLE REGION TIME SERIES
# ============================================================

region_example = df["country_region"].iloc[0]
region_df = df[df["country_region"] == region_example]

plt.figure()
plt.plot(region_df["year"], region_df["incidence"], label="Incidence")
plt.plot(region_df["year"], region_df["precipitation"], label="Precipitation")
plt.title(f"Time Series Example: {region_example}")
plt.xlabel("Year")
plt.legend()
plt.savefig("sample_region_timeseries.png", dpi=300, bbox_inches="tight")
plt.close()

print("\n✅ EDA completed. All figures saved as PNG.")