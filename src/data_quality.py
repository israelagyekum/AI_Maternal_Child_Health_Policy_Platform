
import pandas as pd

print("Loading panel dataset...")

df = pd.read_csv("data/processed/mch_panel_data.csv")

print("\nDataset Overview")
print("------------------")
print("Shape:", df.shape)

print("\nMissing Values Per Column")
print("---------------------------")
print(df.isna().sum())

print("\nMissing Percentage Per Column")
print("-------------------------------")
missing_pct = (df.isna().mean() * 100).round(2)
print(missing_pct)

print("\nPanel Balance Check")
print("--------------------")
years_per_country = df.groupby("country")["year"].nunique()
print("Minimum years per country:", years_per_country.min())
print("Maximum years per country:", years_per_country.max())

print("\nDescriptive Statistics")
print("-----------------------")
print(df.describe())