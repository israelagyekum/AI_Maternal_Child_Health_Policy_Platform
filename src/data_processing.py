
import pandas as pd

print("Loading raw data...")

# Load raw data
df = pd.read_csv("data/raw/world_bank_mch_data.csv")

print("Pivoting data to panel format...")

# Pivot to wide format
df_pivot = df.pivot_table(
    index=["country", "country_code", "year"],
    columns="indicator",
    values="value"
).reset_index()

# Rename columns
df_pivot = df_pivot.rename(columns={
    "SH.STA.MMRT": "maternal_mortality",
    "SH.XPD.CHEX.PC.CD": "health_expenditure_per_capita",
    "SP.DYN.TFRT.IN": "fertility_rate",
    "NY.GDP.PCAP.CD": "gdp_per_capita",
    "SE.SEC.ENRR.FE": "female_secondary_enrollment",
    "SH.MED.PHYS.ZS": "physicians_per_1000"
})

# Convert year to int
df_pivot["year"] = df_pivot["year"].astype(int)

# Save processed data
df_pivot.to_csv("data/processed/mch_panel_data.csv", index=False)

print("Panel dataset created successfully.")
print("Dataset shape:", df_pivot.shape)