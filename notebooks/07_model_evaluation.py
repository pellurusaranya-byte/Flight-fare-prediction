
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load dataset
df = pd.read_csv(PROJECT_ROOT / "data" / "Clean_Dataset.csv")

# Use the same sample as model comparison
df = df.sample(n=50000, random_state=42)

# Remove unnecessary columns
df = df.drop(columns=["Unnamed: 0", "flight"])

# Separate features and target
X = df.drop(columns=["price"])
y = df["price"]

# Split data exactly as before
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained Random Forest model
model_path = PROJECT_ROOT / "models" / "random_forest_sample.joblib"
model = joblib.load(model_path)

# Predict prices
y_pred = model.predict(X_test)

# Calculate errors
errors = y_test.values - y_pred

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nRandom Forest Evaluation")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Create results dataframe
evaluation_df = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred,
    "Error": errors,
    "Absolute Error": np.abs(errors)
})

# 1. Actual vs Predicted Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3)

# Perfect prediction reference line
min_price = min(y_test.min(), y_pred.min())
max_price = max(y_test.max(), y_pred.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)

plt.xlabel("Actual Price (₹)")
plt.ylabel("Predicted Price (₹)")
plt.title("Actual vs Predicted Flight Prices")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "results" / "actual_vs_predicted.png", dpi=300)
plt.close()

# 2. Prediction Error Distribution
plt.figure(figsize=(8, 6))
sns.histplot(errors, bins=50, kde=True)

plt.xlabel("Prediction Error (₹)")
plt.ylabel("Number of Flights")
plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "results" / "error_distribution.png", dpi=300)
plt.close()

# 3. Actual Price vs Absolute Error
plt.figure(figsize=(8, 6))
plt.scatter(y_test, np.abs(errors), alpha=0.3)

plt.xlabel("Actual Price (₹)")
plt.ylabel("Absolute Prediction Error (₹)")
plt.title("Prediction Error vs Actual Flight Price")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "results" / "error_vs_price.png", dpi=300)
plt.close()

# Save evaluation data
evaluation_df.to_csv(
    PROJECT_ROOT / "results" / "prediction_errors.csv",
    index=False
)

print("\nEvaluation graphs saved successfully!")
print("Evaluation data saved successfully!")

