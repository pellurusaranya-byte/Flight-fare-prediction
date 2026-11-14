
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Set project folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 2. Load dataset
df = pd.read_csv(PROJECT_ROOT / "data" / "Clean_Dataset.csv")

# Use the same 50,000-row sample as our earlier comparison
df = df.sample(n=50000, random_state=42).copy()


# 3. Create route groups
# Both directions of the same city pair belong to one group.
df["route_group"] = df.apply(
    lambda row: "_".join(
        sorted([row["source_city"], row["destination_city"]])
    ),
    axis=1
)


# 4. Separate input features and target
df = df.drop(columns=["Unnamed: 0", "flight"])

X = df.drop(columns=["price", "route_group"])
y = df["price"]
groups = df["route_group"]


# 5. Define categorical and numerical columns
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


# 6. Split by route, not by individual row
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

train_routes = set(groups.iloc[train_indices])
test_routes = set(groups.iloc[test_indices])

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training routes:", len(train_routes))
print("Testing routes:", len(test_routes))
print("Routes shared between training and testing:",
      len(train_routes.intersection(test_routes)))


# 7. Function to create preprocessing
def create_preprocessor(dense=False):

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=not dense
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


# 8. Define models
models = {
    "Linear Regression": Pipeline([
        ("preprocessor", create_preprocessor()),
        ("regressor", LinearRegression())
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", create_preprocessor()),
        ("regressor", RandomForestRegressor(
            n_estimators=30,
            max_depth=15,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("preprocessor", create_preprocessor(dense=True)),
        ("regressor", HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.1,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42
        ))
    ])
}


# 9. Train and evaluate each model
results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4)
    })


# 10. Display results
results_df = pd.DataFrame(results)

print("\n========== ROBUST VALIDATION RESULTS ==========")
print(results_df.to_string(index=False))


# 11. Save results
results_df.to_csv(
    PROJECT_ROOT / "results" / "robust_validation.csv",
    index=False
)

print("\nResults saved successfully!")
