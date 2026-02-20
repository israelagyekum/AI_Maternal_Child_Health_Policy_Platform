
import requests
import pandas as pd

# Indicators
indicators = [
    "SH.STA.MMRT",
    "SH.XPD.CHEX.PC.CD",
    "SP.DYN.TFRT.IN",
    "NY.GDP.PCAP.CD",
    "SE.SEC.ENRR.FE",
    "SH.MED.PHYS.ZS"
]

all_data = []

print("Downloading data from World Bank API...")

for indicator in indicators:
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}?date=2000:2022&format=json&per_page=20000"
    
    response = requests.get(url)
    data = response.json()

    if len(data) > 1:
        records = data[1]
        for entry in records:
            all_data.append({
                "country": entry["country"]["value"],
                "country_code": entry["country"]["id"],
                "year": entry["date"],
                "indicator": indicator,
                "value": entry["value"]
            })

df = pd.DataFrame(all_data)

df.to_csv("data/raw/world_bank_mch_data.csv", index=False)

print("World Bank data downloaded successfully.")