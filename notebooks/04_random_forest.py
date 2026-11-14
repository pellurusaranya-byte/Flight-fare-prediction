
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# STEP 1: LOAD DATASET

PROJECT_ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(PROJECT_ROOT / "data" / "Clean_Dataset.csv")

# Use a sample of 50,000 rows to reduce training time
df = df.sample(n=50000, random_state=42)

df = df.drop(columns=["Unnamed: 0", "flight"])


# STEP 2: SEPARATE FEATURES AND TARGET

X = df.drop(columns=["price"])
y = df["price"]


# STEP 3: IDENTIFY COLUMN TYPES

categorical_columns = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class"
]

numerical_columns = [
    "duration",
    "days_left"
]


# STEP 4: SPLIT DATA

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# STEP 5: PREPROCESS DATA

preprocessor = ColumnTransformer(
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


# STEP 6: CREATE RANDOM FOREST MODEL

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=20,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=2
        ))
    ]
)


# STEP 7: TRAIN MODEL

print("Training Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed!")


# STEP 8: MAKE PREDICTIONS

y_pred = model.predict(X_test)


# STEP 9: EVALUATE MODEL

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("\nRandom Forest Results")

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))


# STEP 10: COMPARE ACTUAL AND PREDICTED PRICES

comparison = pd.DataFrame({
    "Actual Price": y_test.iloc[:10].values,
    "Predicted Price": np.round(y_pred[:10], 2)
})

print("\nActual vs Predicted Prices:")
print(comparison)


# STEP 11: SAVE MODEL

model_path = PROJECT_ROOT / "models" / "random_forest.joblib"

joblib.dump(model, model_path)

print("\nModel saved successfully at:", model_path)
