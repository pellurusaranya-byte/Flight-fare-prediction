import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

df = pd.read_csv(PROJECT_ROOT / "data" / "Clean_Dataset.csv")

df = df.drop(columns=["Unnamed: 0", "flight"])

X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = joblib.load(
    PROJECT_ROOT / "models" / "gradient_boosting_final.joblib"
)

predictions = model.predict(X_test)

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions,
    "Class": X_test["class"]
})

for class_name in ["Economy", "Business"]:

    subset = results[results["Class"] == class_name]

    mae = mean_absolute_error(
        subset["Actual"], subset["Predicted"]
    )

    rmse = np.sqrt(mean_squared_error(
        subset["Actual"], subset["Predicted"]
    ))

    r2 = r2_score(
        subset["Actual"], subset["Predicted"]
    )

    bias = (
        subset["Predicted"] - subset["Actual"]
    ).mean()

    print(f"\n{class_name} Results")
    print("Records:", len(subset))
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))
    print("Average Prediction Bias:", round(bias, 2))