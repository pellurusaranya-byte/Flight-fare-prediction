
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load dataset
df = pd.read_csv(PROJECT_ROOT / "data" / "Clean_Dataset.csv")

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

numerical_columns = [
    "duration",
    "days_left"
]

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create preprocessing steps
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

# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same transformations to testing data
X_test_processed = preprocessor.transform(X_test)

# Display results
print("Original dataset shape:", df.shape)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("Processed training data shape:", X_train_processed.shape)
print("Processed testing data shape:", X_test_processed.shape)

print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)

print("\nPreprocessing completed successfully!")
