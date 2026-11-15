
import pandas as pd
import joblib
from pathlib import Path


# 1. Locate project folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 2. Load the saved model
model_path = PROJECT_ROOT / "models" / "gradient_boosting_final.joblib"

model = joblib.load(model_path)

print("Final model loaded successfully!")


# 3. Create an example flight search
flight = pd.DataFrame([{
    "airline": "Indigo",
    "source_city": "Delhi",
    "departure_time": "Morning",
    "stops": "zero",
    "destination_city": "Mumbai",
    "class": "Economy",
    "duration": 2.0,
    "days_left": 15
}])


# 4. Predict the flight price
predicted_price = model.predict(flight)[0]


# 5. Display the result
print("\n========== FLYWISE PRICE PREDICTION ==========")

print("Airline:", flight.iloc[0]["airline"])
print("Route:", flight.iloc[0]["source_city"], "to",
      flight.iloc[0]["destination_city"])
print("Class:", flight.iloc[0]["class"])
print("Duration:", flight.iloc[0]["duration"], "hours")
print("Days left:", flight.iloc[0]["days_left"])

print("\nPredicted Flight Price: ₹", round(predicted_price, 2))
