import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import numpy as np
try:
    model_data = joblib.load("models/maternal_model.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please train the model first.")
    st.stop()
# ==========================================================
# INSTITUTIONAL VISUAL THEME
# ==========================================================

INSTITUTIONAL_BLUE = "#1F4E79"
INSTITUTIONAL_GREEN = "#2E8B57"
INSTITUTIONAL_RED = "#B22222"
INSTITUTIONAL_GREY = "#D9D9D9"
DARK_BACKGROUND = "#0B1F3A"

import plotly.io as pio

pio.templates["institutional"] = pio.templates["plotly_dark"]

pio.templates["institutional"].layout.update(
    plot_bgcolor=DARK_BACKGROUND,
    paper_bgcolor=DARK_BACKGROUND,
    font=dict(color="white", size=13),
    title=dict(x=0.02)
)

pio.templates.default = "institutional"    
 
# ==========================================================
# LOAD DATA + MODEL (CACHED)
# ==========================================================

# ==========================================================
# INSTITUTIONAL PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Maternal & Child Health Policy Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# Institutional Header
# ----------------------------------------------------------

st.markdown("""
# Maternal & Child Health Policy Intelligence Platform  
### Econometric & Machine Learning Decision Support System  

**Data Source:** World Bank (2000–2022)  
**Core Model:** Two-Way Fixed Effects Panel Regression (Clustered SE)  
**ML Benchmarking:** Random Forest & Gradient Boosting  
""")

st.caption("Version 1.0 — Institutional Release")

st.markdown("---")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/mch_panel_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/maternal_model.pkl")

df = load_data()
model_data = load_model()

df = df.dropna()

# ----------------------------------------------------------
# Executive Overview
# ----------------------------------------------------------

st.markdown("""
### Executive Overview

This platform integrates econometric modelling and machine learning 
to support evidence-based maternal health policy analysis.  

It enables:

• Structural evaluation of long-run determinants  
• Predictive benchmarking using AI models  
• Counterfactual policy simulation  
• Transparent interpretability of model outputs  

The system is designed for analytical support and does not replace 
formal policy decision-making processes.
""")

st.markdown("---")



# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Maternal Health Policy Econometric Platform",
    layout="wide"
)



# ==========================================================
# LOAD DATA + MODELS
# ==========================================================
df = pd.read_csv("data/processed/mch_panel_data.csv")

model_data = joblib.load("models/maternal_model.pkl")

global_model = model_data["model"]
global_r2 = model_data["r2"]
global_rmse = model_data["rmse"]
global_mae = model_data["mae"]

rf_model = model_data["rf_model"]
rf_r2 = model_data["rf_r2"]
rf_rmse = model_data["rf_rmse"]
rf_mae = model_data["rf_mae"]

gb_model = model_data["gb_model"]
gb_r2 = model_data["gb_r2"]
gb_rmse = model_data["gb_rmse"]
gb_mae = model_data["gb_mae"]

policy_vars = [
    "gdp_per_capita",
    "fertility_rate",
    "health_expenditure_per_capita",
    "female_secondary_enrollment"
]

# ==========================================================
# SIDEBAR FILTERS
# ==========================================================
st.sidebar.header("Filter Options")

selected_country = st.sidebar.selectbox(
    "Select Country",
    sorted(df["country"].unique())
)

selected_year = st.sidebar.selectbox(
    "Select Year",
    sorted(df["year"].unique())
)

filtered_df = df[df["country"] == selected_country]
latest_data = filtered_df[filtered_df["year"] == selected_year]

if latest_data.empty:
    st.warning("No data available for selected year.")
    st.stop()

# ==========================================================
# KPI SECTION
# ==========================================================
st.markdown("## Key Indicators")

row = latest_data.iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric("Maternal Mortality", round(row["maternal_mortality"], 2))
col2.metric("Fertility Rate", round(row["fertility_rate"], 2))
col3.metric("GDP per Capita", round(row["gdp_per_capita"], 2))



# ==========================================================
# POLICY SHOCK SIMULATION
# ==========================================================
st.markdown("## Policy Scenario Simulation")

gdp_base = row["gdp_per_capita"]
fertility_base = row["fertility_rate"]
health_base = row["health_expenditure_per_capita"]
education_base = row["female_secondary_enrollment"]

col1, col2 = st.columns(2)

with col1:
    gdp_shock = st.slider("GDP Change (%)", -20, 20, 0)
    health_shock = st.slider("Health Expenditure Change (%)", -30, 50, 0)

with col2:
    fertility_shock = st.slider("Fertility Change (%)", -50, 20, 0)
    education_shock = st.slider("Education Change (%)", -20, 50, 0)

# Apply shocks
gdp_new = gdp_base * (1 + gdp_shock / 100)
health_new = health_base * (1 + health_shock / 100)
fertility_new = fertility_base * (1 + fertility_shock / 100)
education_new = education_base * (1 + education_shock / 100)

# ==========================================================
# PREDICTION INPUTS
# ==========================================================

shock_input = pd.DataFrame({
    "gdp_per_capita": [gdp_new],
    "fertility_rate": [fertility_new],
    "health_expenditure_per_capita": [health_new],
    "female_secondary_enrollment": [education_new],
    "country": [selected_country],
    "year": [selected_year]
})

shock_input["country"] = shock_input["country"].astype("category")
shock_input["year"] = shock_input["year"].astype("category")

# Panel Prediction
panel_prediction = global_model.predict(shock_input)[0]

# ML Predictions
# Prepare ML input safely
ml_input = shock_input[policy_vars].copy()

# Force numeric conversion
ml_input = ml_input.apply(pd.to_numeric, errors="coerce")

# Check for missing values
if ml_input.isnull().any().any():
    st.error("ML prediction failed: Missing values detected in inputs.")
    rf_prediction = None
    gb_prediction = None
else:
    rf_prediction = rf_model.predict(ml_input)[0]
    gb_prediction = gb_model.predict(ml_input)[0]

# Baseline for impact comparison
baseline_input = pd.DataFrame({
    "gdp_per_capita": [gdp_base],
    "fertility_rate": [fertility_base],
    "health_expenditure_per_capita": [health_base],
    "female_secondary_enrollment": [education_base],
    "country": [selected_country],
    "year": [selected_year]
})

baseline_input["country"] = baseline_input["country"].astype("category")
baseline_input["year"] = baseline_input["year"].astype("category")

baseline_prediction = global_model.predict(baseline_input)[0]
impact = baseline_prediction - panel_prediction

# ==========================================================
# POLICY IMPACT DISPLAY
# ==========================================================
# ----------------------------------------------------------
# Policy Impact Projection
# ----------------------------------------------------------

# ----------------------------------------------------------
# Policy Impact Projection
# ----------------------------------------------------------

st.markdown("### Policy Impact Projection")

col1, col2, col3 = st.columns(3)

col1.metric("Baseline (Panel FE)", round(baseline_prediction, 2))
col2.metric("Post-Policy (Panel FE)", round(panel_prediction, 2))
col3.metric("Projected Reduction", round(impact, 2))

# --- Create comparison dataframe BEFORE plotting ---
comparison_df = pd.DataFrame({
    "Scenario": ["Baseline", "Post-Policy"],
    "Mortality": [baseline_prediction, panel_prediction]
})

fig_policy = px.bar(
    comparison_df,
    x="Scenario",
    y="Mortality",
    color="Scenario",
    color_discrete_map={
        "Baseline": INSTITUTIONAL_BLUE,
        "Post-Policy": INSTITUTIONAL_GREEN
    },
    title="Policy Intervention Impact"
)

fig_policy.update_layout(showlegend=False)

st.plotly_chart(fig_policy, use_container_width=True)

# ==========================================================
# HYBRID PREDICTION COMPARISON
# ==========================================================
st.markdown("### Hybrid Prediction Comparison")

col1, col2, col3 = st.columns(3)

col1.metric("Panel FE Prediction", round(panel_prediction, 2))

if rf_prediction is not None:
    col2.metric("Random Forest Prediction", round(rf_prediction, 2))
else:
    col2.metric("Random Forest Prediction", "Unavailable")

if gb_prediction is not None:
    col3.metric("Gradient Boosting Prediction", round(gb_prediction, 2))
else:
    col3.metric("Gradient Boosting Prediction", "Unavailable")

# ==========================================================
# TRENDS
# ==========================================================
st.markdown("## Trends Over Time")

fig_trend = px.line(
    filtered_df,
    x="year",
    y="maternal_mortality",
    title="Maternal Mortality Trend"
)

st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# ==========================================================
# STRUCTURAL COEFFICIENT INTERPRETATION
# ==========================================================
st.markdown("## Structural Coefficient Interpretation")

coef_df = pd.DataFrame({
    "Variable": policy_vars,
    "Coefficient": [global_model.params[var] for var in policy_vars]
})

fig_coef = px.bar(
    coef_df,
    x="Variable",
    y="Coefficient",
    title="Within-Country Marginal Effects"
)

st.plotly_chart(fig_coef, use_container_width=True)

# ==========================================================
# MODEL PERFORMANCE COMPARISON
# ==========================================================
st.markdown("## Model Performance Comparison")

performance_df = pd.DataFrame({
    "Model": ["Panel Fixed Effects", "Random Forest", "Gradient Boosting"],
    "R²": [global_r2, rf_r2, gb_r2],
    "RMSE": [global_rmse, rf_rmse, gb_rmse],
    "MAE": [global_mae, rf_mae, gb_mae]
})

st.dataframe(performance_df.round(3))

# ==========================================================
# INFERENCE TABLE
# ==========================================================
st.markdown("## Inference (Clustered Standard Errors)")

summary_df = pd.DataFrame({
    "Coefficient": global_model.params[policy_vars],
    "Std Error (Clustered)": global_model.bse[policy_vars],
    "t-Statistic": global_model.tvalues[policy_vars],
    "p-Value": global_model.pvalues[policy_vars]
}).round(4)

st.dataframe(summary_df)

# ==========================================================
# MODEL DIAGNOSTICS
# ==========================================================
st.markdown("## Model Diagnostics")

panel_df = df.dropna(subset=policy_vars + ["maternal_mortality"]).copy()
panel_df["country"] = panel_df["country"].astype("category")
panel_df["year"] = panel_df["year"].astype("category")

panel_df["Predicted"] = global_model.predict(panel_df)

fig_actual = px.scatter(
    panel_df,
    x="maternal_mortality",
    y="Predicted",
    opacity=0.6,
    title="Actual vs Predicted (Panel Fixed Effects)"
)

fig_actual.add_shape(
    type="line",
    x0=panel_df["maternal_mortality"].min(),
    y0=panel_df["maternal_mortality"].min(),
    x1=panel_df["maternal_mortality"].max(),
    y1=panel_df["maternal_mortality"].max(),
    line=dict(dash="dash")
)

st.plotly_chart(fig_actual, use_container_width=True)

st.caption("Model specification: Two-way fixed effects panel regression with clustered standard errors by country.")

# ==========================================================
# EXPLAINABLE AI (SHAP)
# ==========================================================

import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": DARK_BACKGROUND,
    "axes.facecolor": DARK_BACKGROUND,
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "text.color": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "axes.titlecolor": "white",
    "axes.grid": False,
})

st.markdown("## Explainable AI (SHAP Interpretation)")

# Use same ML input as prediction
ml_input = shock_input[policy_vars].copy()
ml_input = ml_input.apply(pd.to_numeric, errors="coerce")

if not ml_input.isnull().any().any():

    st.markdown("### Random Forest Explanation")

    rf_explainer = shap.TreeExplainer(rf_model)
    rf_shap_values = rf_explainer.shap_values(ml_input)

    fig, ax = plt.subplots()
    shap.bar_plot(rf_shap_values[0], feature_names=policy_vars)
    st.pyplot(fig)

    st.markdown("### Gradient Boosting Explanation")

    gb_explainer = shap.TreeExplainer(gb_model)
    gb_shap_values = gb_explainer.shap_values(ml_input)

    fig2, ax2 = plt.subplots()
    shap.bar_plot(gb_shap_values[0], feature_names=policy_vars)
    st.pyplot(fig2)

else:
    st.warning("SHAP explanation unavailable due to missing inputs.")

    # ==========================================================
# GLOBAL SHAP SUMMARY
# ==========================================================

st.markdown("### Global SHAP Summary (Random Forest)")

global_ml_df = df.dropna(subset=policy_vars).copy()

X_global = global_ml_df[policy_vars]

rf_explainer_global = shap.TreeExplainer(rf_model)
rf_shap_global = rf_explainer_global.shap_values(X_global)

fig_summary, ax_summary = plt.subplots()
shap.summary_plot(
    rf_shap_global,
    X_global,
    feature_names=policy_vars,
    show=False
)
st.pyplot(fig_summary)

st.markdown("### Local SHAP Waterfall Plot (Policy Scenario)")

if not ml_input.isnull().any().any():

    rf_explainer = shap.TreeExplainer(rf_model)
    rf_shap_values = rf_explainer.shap_values(ml_input)

    shap_values_single = rf_shap_values[0]

    base_value = rf_explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[0]

    explanation = shap.Explanation(
        values=shap_values_single,
        base_values=base_value,
        data=ml_input.iloc[0],
        feature_names=policy_vars
    )

    # Strong contrast settings
    plt.style.use("default")
    fig_waterfall, ax = plt.subplots(figsize=(10, 6), dpi=120)
    fig_waterfall.patch.set_facecolor("white")

    shap.plots.waterfall(explanation, show=False)

    # Improve readability
    ax.tick_params(labelsize=11)
    ax.title.set_size(14)

    st.pyplot(fig_waterfall)

else:
    st.warning("SHAP explanation unavailable due to missing inputs.")
    
    # ==========================================================
# PARTIAL DEPENDENCE PLOTS
# ==========================================================

st.markdown("## Global Nonlinear Effects (Model-Level)")

fig_pdp, ax_pdp = plt.subplots(figsize=(10, 6))

PartialDependenceDisplay.from_estimator(
    rf_model,
    X_global,
    policy_vars,
    ax=ax_pdp
)

st.pyplot(fig_pdp)

st.caption(
    "These curves represent global nonlinear relationships learned by the Random Forest model "
    "across all countries (2000–2022). They are not country-specific."
)

# ----------------------------------------------------------
# Governance & Transparency
# ----------------------------------------------------------

st.markdown("---")
st.markdown("""
### Governance & Transparency

Data Source: World Bank (2000–2022)  
Model Framework: Two-Way Fixed Effects with Clustered Standard Errors  
Machine Learning Benchmark: Random Forest & Gradient Boosting  

This platform is intended for analytical and research purposes. 
Results should be interpreted within broader contextual and institutional frameworks.
""")

st.markdown("---")
st.caption(
    "Maternal & Child Health Policy Intelligence Platform | "
    "Econometric & AI Decision Support | Institutional Release"
)