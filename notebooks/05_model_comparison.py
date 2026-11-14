
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

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

# Feature categories
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_columns
            )
        ]
    )


# Define models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=30,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.1,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42
    )
}

results = []

# Train and evaluate each model
for model_name, regressor in models.items():

    print(f"\nTraining {model_name}...")

    if model_name == "Gradient Boosting":
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ),
                    categorical_columns
                ),
                (
                    "numerical",
                    StandardScaler(),
                    numerical_columns
                )
            ]
        )
    else:
        preprocessor = create_preprocessor()

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ]
    )

    model.fit(X_train, y_train)

    print(f"{model_name} training completed!")

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4)
    })

    # Save trained model
    model_filename = model_name.lower().replace(" ", "_") + "_final.joblib"

    joblib.dump(
        model,
        PROJECT_ROOT / "models" / model_filename
    )


# Create comparison table
results_df = pd.DataFrame(results)

print("\n========== FINAL MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# Save comparison table
results_df.to_csv(
    PROJECT_ROOT / "results" / "final_model_comparison.csv",
    index=False
)

# Create comparison graph
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

metrics = ["MAE", "RMSE", "R2 Score"]
titles = ["Mean Absolute Error", "Root Mean Squared Error", "R2 Score"]

for ax, metric, title in zip(axes, metrics, titles):

    ax.bar(results_df["Model"], results_df[metric])

    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=25)

plt.suptitle("Comparison of Flight Price Prediction Models")
plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "results" / "final_model_comparison.png",
    dpi=300
)

plt.close()

print("\nFinal comparison table and graph saved successfully!")
print("All three trained models have been saved!")
