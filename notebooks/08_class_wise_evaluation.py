
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load dataset
df = pd.read_csv(PROJECT_ROOT / "data" / "Clean_Dataset.csv")

# Select the same 50,000 records
df = df.sample(n=50000, random_state=42)

# Remove unnecessary columns
df = df.drop(columns=["Unnamed: 0", "flight"])

# Separate features and target
X = df.drop(columns=["price"])
y = df["price"]

# Split data using the same settings
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load saved Random Forest model
model_path = PROJECT_ROOT / "models" / "random_forest_sample.joblib"
model = joblib.load(model_path)

# Predict prices
y_pred = model.predict(X_test)

# Create evaluation DataFrame
evaluation_df = pd.DataFrame({
    "Class": X_test["class"].values,
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

# Calculate prediction errors
evaluation_df["Error"] = (
    evaluation_df["Actual Price"] - evaluation_df["Predicted Price"]
)

evaluation_df["Absolute Error"] = evaluation_df["Error"].abs()

# Calculate metrics separately for each class
results = []

for ticket_class in evaluation_df["Class"].unique():

    class_data = evaluation_df[
        evaluation_df["Class"] == ticket_class
    ]

    actual = class_data["Actual Price"]
    predicted = class_data["Predicted Price"]

    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(
    np.abs((actual.values - predicted.values) / actual.values)
) * 100
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    results.append({
        "Class": ticket_class,
        "Number of Flights": len(class_data),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4),
        "MAPE (%)": round(mape, 2)
    })

# Display results
results_df = pd.DataFrame(results)

print("\nCLASS-WISE MODEL EVALUATION")
print(results_df.to_string(index=False))

# Save class-wise metrics
results_df.to_csv(
    PROJECT_ROOT / "results" / "class_wise_metrics.csv",
    index=False
)

# Plot absolute prediction errors by class
plt.figure(figsize=(9, 6))

sns.boxplot(
    data=evaluation_df,
    x="Class",
    y="Absolute Error"
)

plt.xlabel("Ticket Class")
plt.ylabel("Absolute Prediction Error (₹)")
plt.title("Prediction Errors by Ticket Class")

plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "results" / "class_wise_errors.png",
    dpi=300
)

plt.close()

print("\nClass-wise evaluation completed!")
print("Metrics and graph saved successfully!")
