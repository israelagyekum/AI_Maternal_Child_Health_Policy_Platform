import pandas as pd
import statsmodels.formula.api as smf
import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_PATH = "data/processed/mch_panel_data.csv"
MODEL_PATH = "models/maternal_model.pkl"

FEATURES = [
    "gdp_per_capita",
    "fertility_rate",
    "health_expenditure_per_capita",
    "female_secondary_enrollment"
]

TARGET = "maternal_mortality"


# ==========================================================
# TRAIN HYBRID MODEL (Econometric + ML)
# ==========================================================

def train_and_save_model():
    """
    Trains:
    - Two-way Fixed Effects Panel Regression (clustered SE)
    - Random Forest
    - Gradient Boosting
    
    Saves hybrid model dictionary to disk.
    """

    # -----------------------------
    # Load & Prepare Data
    # -----------------------------
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=FEATURES + [TARGET])

    df["country"] = df["country"].astype("category")
    df["year"] = df["year"].astype("category")

    # -----------------------------
    # PANEL MODEL
    # -----------------------------
    panel_model = smf.ols(
        formula=f"""
        {TARGET} ~ 
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

    panel_r2 = panel_model.rsquared
    panel_rmse = np.sqrt(mean_squared_error(
        df[TARGET],
        panel_model.fittedvalues
    ))
    panel_mae = mean_absolute_error(
        df[TARGET],
        panel_model.fittedvalues
    )

    # -----------------------------
    # MACHINE LEARNING MODELS
    # -----------------------------
    X_ml = df[FEATURES]
    y_ml = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X_ml, y_ml, test_size=0.2, random_state=42
    )

    # Random Forest
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

    # Gradient Boosting
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

    # -----------------------------
    # SAVE HYBRID DICTIONARY
    # -----------------------------
    hybrid_model = {
        "panel_model": panel_model,
        "panel_metrics": {
            "r2": panel_r2,
            "rmse": panel_rmse,
            "mae": panel_mae
        },
        "rf_model": rf,
        "rf_metrics": {
            "r2": rf_r2,
            "rmse": rf_rmse,
            "mae": rf_mae
        },
        "gb_model": gb,
        "gb_metrics": {
            "r2": gb_r2,
            "rmse": gb_rmse,
            "mae": gb_mae
        }
    }

    joblib.dump(hybrid_model, MODEL_PATH)

    print("Hybrid Econometric + ML model saved successfully.")
    return hybrid_model


# ==========================================================
# LOAD MODEL (Used by Dashboard)
# ==========================================================

def train_model():
    """
    Loads the previously trained hybrid model.
    If model file does not exist, trains and saves automatically.
    """
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return train_and_save_model()