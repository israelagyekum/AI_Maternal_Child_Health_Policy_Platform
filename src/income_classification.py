import requests
import pandas as pd

print("Downloading income classification...")

url = "https://api.worldbank.org/v2/country?per_page=500&format=json"

response = requests.get(url)
data = response.json()

records = data[1]

income_data = []

for country in records:
    income_data.append({
        "country": country["name"],
        "country_code": country["id"],
        "income_group": country["incomeLevel"]["value"]
    })

df_income = pd.DataFrame(income_data)

df_income.to_csv("data/raw/world_bank_income_classification.csv", index=False)

print("Income classification downloaded successfully.")