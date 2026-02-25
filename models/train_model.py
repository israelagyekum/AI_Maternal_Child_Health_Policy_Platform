import pandas as pd
import statsmodels.formula.api as smf
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("data/processed/mch_panel_data.csv")

df = df.dropna(subset=[
    "gdp_per_capita",
    "fertility_rate",
    "health_expenditure_per_capita",
    "female_secondary_enrollment",
    "maternal_mortality"
])

df["country"] = df["country"].astype("category")
df["year"] = df["year"].astype("category")

# ==========================================
# PANEL MODEL (Two-Way Fixed Effects)
# ==========================================

panel_model = smf.ols(
    formula="""
    maternal_mortality ~ 
    gdp_per_capita +
    fertility_rate +
    health_expenditure_per_capita +
    female_secondary_enrollment +
    C(country) +
    C(year)
    """,
    data=df
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["country"]}
)

# Panel metrics (in-sample structural fit)
panel_r2 = panel_model.rsquared
panel_rmse = np.sqrt(mean_squared_error(
    df["maternal_mortality"],
    panel_model.fittedvalues
))
panel_mae = mean_absolute_error(
    df["maternal_mortality"],
    panel_model.fittedvalues
)

print("\nPanel Fixed Effects Performance")
print("R2:", panel_r2)
print("RMSE:", panel_rmse)
print("MAE:", panel_mae)

# ==========================================
# MACHINE LEARNING BENCHMARK MODELS
# ==========================================

X_ml = df[[
    "gdp_per_capita",
    "fertility_rate",
    "health_expenditure_per_capita",
    "female_secondary_enrollment"
]]

y_ml = df["maternal_mortality"]

X_train, X_test, y_train, y_test = train_test_split(
    X_ml, y_ml, test_size=0.2, random_state=42
)

# ---------- Random Forest ----------
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    random_state=42
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)

print("\nRandom Forest Performance")
print("R2:", rf_r2)
print("RMSE:", rf_rmse)
print("MAE:", rf_mae)

# ---------- Gradient Boosting ----------
gb = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

gb_r2 = r2_score(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_mae = mean_absolute_error(y_test, gb_pred)

print("\nGradient Boosting Performance")
print("R2:", gb_r2)
print("RMSE:", gb_rmse)
print("MAE:", gb_mae)

# ==========================================
# SAVE EVERYTHING
# ==========================================

joblib.dump({
    "model": panel_model,
    "r2": panel_r2,
    "rmse": panel_rmse,
    "mae": panel_mae,
    "rf_model": rf,
    "rf_r2": rf_r2,
    "rf_rmse": rf_rmse,
    "rf_mae": rf_mae,
    "gb_model": gb,
    "gb_r2": gb_r2,
    "gb_rmse": gb_rmse,
    "gb_mae": gb_mae
}, "models/maternal_model.pkl")

print("\nHybrid Econometric + ML model saved successfully.")