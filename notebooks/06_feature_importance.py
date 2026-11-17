
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
(PROJECT_ROOT / "results" / "exploratory").mkdir(parents=True, exist_ok=True)  # superseded experiments write here

# Load the trained Random Forest model
model_path = PROJECT_ROOT / "models" / "random_forest_sample.joblib"
model = joblib.load(model_path)

# Extract the trained Random Forest regressor
random_forest = model.named_steps["regressor"]

# Get feature names after preprocessing
preprocessor = model.named_steps["preprocessor"]
feature_names = preprocessor.get_feature_names_out()

# Get feature importance scores
importance_scores = random_forest.feature_importances_

# Create a DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_scores
})

# Sort features by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Display top 15 features
top_features = importance_df.head(15)

print("\nTop 15 Important Features:")
print(top_features.to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Top 15 Features Influencing Flight Prices")

plt.gca().invert_yaxis()
plt.tight_layout()

# Save graph
graph_path = PROJECT_ROOT / "results" / "exploratory" / "feature_importance.png"
plt.savefig(graph_path, dpi=300)
plt.close()

# Save feature importance data
csv_path = PROJECT_ROOT / "results" / "exploratory" / "feature_importance.csv"
importance_df.to_csv(csv_path, index=False)

print("\nFeature importance graph saved at:", graph_path)
print("Feature importance data saved at:", csv_path)
