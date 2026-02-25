import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

print("Loading dataset...")

df = pd.read_csv("data/processed/mch_panel_data.csv")

# Keep relevant variables
df_model = df[[
    "country",
    "year",
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

print("Running Fixed Effects regression...")

# Fixed effects model using country and year dummies
print("Running Fixed Effects regression with clustered SE...")

model = smf.ols(
    formula="""
    log_mmr ~ log_gdp + log_health_exp + fertility_rate + female_secondary_enrollment
    + C(country) + C(year)
    """,
    data=df_model
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df_model["country"]}
)

print(model.summary())