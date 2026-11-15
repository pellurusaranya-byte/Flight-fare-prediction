
import pandas as pd
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor


# 1. Find project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 2. Load the complete dataset
df = pd.read_csv(
    PROJECT_ROOT / "data" / "Clean_Dataset.csv"
)

print("Original dataset shape:", df.shape)


# 3. Remove unnecessary columns
df = df.drop(columns=["Unnamed: 0", "flight"])

print("Dataset after removing unnecessary columns:", df.shape)


# 4. Separate input features and target
# arrival_time is NOT used: it is not something a user supplies when searching for a fare.
X = df.drop(columns=["price", "arrival_time"])
y = df["price"]


# 5. Identify categorical and numerical features
categorical_columns = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "destination_city",
    "class"
]

numerical_columns = [
    "duration",
    "days_left"
]


# 6. Create preprocessing
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


# 7. Create the Gradient Boosting model selected by cross-validation in 15_final_evaluation.py
#    (setting B: 15 leaves, learning rate 0.1, 150 iterations)
gradient_boosting = HistGradientBoostingRegressor(
    max_iter=150,
    learning_rate=0.1,
    max_leaf_nodes=15,
    l2_regularization=1.0,
    random_state=42
)


# 8. Combine preprocessing and model
final_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", gradient_boosting)
    ]
)


# 9. Train using the complete dataset
print("\nTraining final Gradient Boosting model...")
print("Training rows:", len(X))

final_model.fit(X, y)

print("Training completed successfully!")


# 10. Save the final model
model_directory = PROJECT_ROOT / "models"
model_directory.mkdir(exist_ok=True)

model_path = model_directory / "gradient_boosting_final.joblib"

joblib.dump(final_model, model_path)

print("\nFinal model saved successfully!")
print("Model location:", model_path)
print("Total training records:", len(X))
