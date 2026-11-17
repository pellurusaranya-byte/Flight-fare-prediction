
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
(PROJECT_ROOT / "results" / "exploratory").mkdir(parents=True, exist_ok=True)  # superseded experiments write here


# 2. Load dataset and take the same sample
df = pd.read_csv(PROJECT_ROOT / "data" / "Clean_Dataset.csv")
df = df.sample(n=50000, random_state=42).copy()


# 3. Group both directions of each city pair together
df["route_group"] = df.apply(
    lambda row: "_".join(
        sorted([row["source_city"], row["destination_city"]])
    ),
    axis=1
)


# 4. Prepare features and target
df = df.drop(columns=["Unnamed: 0", "flight"])

X = df.drop(columns=["price", "route_group"])
y = df["price"]
groups = df["route_group"]


# 5. Feature columns
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


# 6. Create preprocessing function
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


# 7. Define model creation function
def create_models():

    return {
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


# 8. Five-fold group cross-validation
group_kfold = GroupKFold(n_splits=5)

all_results = []

for fold, (train_indices, test_indices) in enumerate(
    group_kfold.split(X, y, groups=groups), start=1
):

    print(f"\n========== FOLD {fold} ==========")

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    train_groups = set(groups.iloc[train_indices])
    test_groups = set(groups.iloc[test_indices])

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Training routes:", len(train_groups))
    print("Testing routes:", len(test_groups))
    print("Shared routes:", len(train_groups.intersection(test_groups)))

    models = create_models()

    for name, model in models.items():

        print(f"Training {name}...")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        all_results.append({
            "Fold": fold,
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2
        })


# 9. Store individual fold results
results_df = pd.DataFrame(all_results)

print("\n========== INDIVIDUAL FOLD RESULTS ==========")
print(results_df.round(2).to_string(index=False))


# 10. Calculate average performance
summary_df = results_df.groupby("Model").agg(
    Mean_MAE=("MAE", "mean"),
    Mean_RMSE=("RMSE", "mean"),
    Mean_R2=("R2 Score", "mean"),
    Std_MAE=("MAE", "std"),
    Std_RMSE=("RMSE", "std"),
    Std_R2=("R2 Score", "std")
).reset_index()

print("\n========== FIVE-FOLD AVERAGE RESULTS ==========")
print(summary_df.round(4).to_string(index=False))


# 11. Save results
results_df.to_csv(
    PROJECT_ROOT / "results" / "exploratory" / "group_cv_fold_results.csv",
    index=False
)

summary_df.to_csv(
    PROJECT_ROOT / "results" / "exploratory" / "group_cv_summary.csv",
    index=False
)

print("\nCross-validation results saved successfully!")
