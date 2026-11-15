
import pandas as pd
import json
from pathlib import Path

# Project directory
ROOT = Path(__file__).resolve().parent

# Read the original dataset
df = pd.read_csv(ROOT / "data" / "Clean_Dataset.csv")

# 1. Average price by airline, class and source city
airline_prices = (
    df.groupby(["class", "source_city", "airline"])["price"]
    .agg(["mean", "count"])
    .reset_index()
)

# 2. Average price by destination, class and source city
destination_prices = (
    df.groupby(["class", "source_city", "destination_city"])["price"]
    .agg(["mean", "count"])
    .reset_index()
)

# 3. Average price according to days left
days_prices = (
    df.groupby(["class", "days_left"])["price"]
    .agg(["mean", "count"])
    .reset_index()
)

# Convert the data into JSON-compatible records
data = {
    "classes": sorted(df["class"].unique().tolist()),

    "source_cities": sorted(
        df["source_city"].unique().tolist()
    ),

    "airline_prices": airline_prices.rename(
        columns={"mean": "avg_price", "count": "flights"}
    ).round(2).to_dict(orient="records"),

    "destination_prices": destination_prices.rename(
        columns={"mean": "avg_price", "count": "flights"}
    ).round(2).to_dict(orient="records"),

    "days_left_prices": days_prices.rename(
        columns={"mean": "avg_price", "count": "flights"}
    ).round(2).to_dict(orient="records")
}

# Create the output directory if necessary
output_dir = ROOT / "docs" / "data"
output_dir.mkdir(parents=True, exist_ok=True)

# Save JSON
output_file = output_dir / "explorer.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, separators=(",", ":"))

print("Fare explorer data exported successfully!")
print("Total records:", len(df))
print("Output file:", output_file)
