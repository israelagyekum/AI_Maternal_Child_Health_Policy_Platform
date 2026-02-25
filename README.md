# Maternal & Child Health Policy Intelligence Platform  
### Econometric & Machine Learning Decision Support System  

---

## Overview

This platform integrates panel econometrics and machine learning to support evidence-based maternal health policy analysis.

It provides:

- Two-way fixed effects panel regression (clustered standard errors)
- Random Forest and Gradient Boosting benchmarking
- Counterfactual policy simulation engine
- Explainable AI (SHAP) interpretability layer
- Global nonlinear response analysis (Partial Dependence)

The system is designed to support analytical evaluation of macroeconomic and social determinants of maternal mortality.

---

## Data Source

World Bank Development Indicators (2000–2022)

---

## Methodological Framework

### Econometric Model
Two-way fixed effects panel regression with clustered standard errors by country.

### Machine Learning Models
- Random Forest Regressor
- Gradient Boosting Regressor

### Interpretability Layers
- Global SHAP importance
- Local SHAP explanations
- Global nonlinear effects (Partial Dependence)

---

## Deployment

Run locally:

streamlit run app/dashboard.py

---

## Governance & Transparency

This platform is intended for analytical and research purposes.  
Results should be interpreted within broader institutional and contextual frameworks.

Version 1.0 — Institutional Release