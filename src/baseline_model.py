
import pandas as pd
import numpy as np
import statsmodels.api as sm

print("Loading dataset...")

df = pd.read_csv("data/processed/mch_panel_data.csv")

# Select relevant columns
df_model = df[[
    "maternal_mortality",
    "gdp_per_capita",
    "health_expenditure_per_capita",
    "fertility_rate",
    "female_secondary_enrollment"
]].copy()

# Drop missing values
df_model = df_model.dropna()

print("Observations after dropping missing:", df_model.shape[0])

# Log transformations
df_model["log_mmr"] = np.log(df_model["maternal_mortality"])
df_model["log_gdp"] = np.log(df_model["gdp_per_capita"])
df_model["log_health_exp"] = np.log(df_model["health_expenditure_per_capita"])

# Define dependent and independent variables
X = df_model[[
    "log_gdp",
    "log_health_exp",
    "fertility_rate",
    "female_secondary_enrollment"
]]

y = df_model["log_mmr"]

# Add constant
X = sm.add_constant(X)

print("Running regression...")

model = sm.OLS(y, X).fit()

print(model.summary())