
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load dataset
df = pd.read_csv(PROJECT_ROOT / "data" / "Clean_Dataset.csv")

# Use the same 50,000 records for fair comparison
df = df.sample(n=50000, random_state=42)

# Remove unnecessary columns
df = df.drop(columns=["Unnamed: 0", "flight"])

# Separate features and target
X = df.drop(columns=["price"])
y = df["price"]

# Identify categorical and numerical columns
categorical_columns = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class"
]

numerical_columns = ["duration", "days_left"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_columns
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        )
    ]
)

# Create Gradient Boosting pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.1,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=42
            )
        )
    ]
)

# Train model
print("Training Gradient Boosting model...")
model.fit(X_train, y_train)

print("Gradient Boosting training completed!")

# Predict prices
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nGradient Boosting Results")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Display sample predictions
comparison = pd.DataFrame({
    "Actual Price": y_test.iloc[:10].values,
    "Predicted Price": np.round(y_pred[:10], 2)
})

print("\nActual vs Predicted Prices:")
print(comparison)

# Save model
model_path = PROJECT_ROOT / "models" / "gradient_boosting_sample.joblib"

joblib.dump(model, model_path)

print("\nModel saved successfully at:", model_path)
