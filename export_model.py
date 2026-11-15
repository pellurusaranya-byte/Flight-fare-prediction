
import joblib
import json
import numpy as np
from pathlib import Path

# Load the trained FlyWise model
model = joblib.load("models/gradient_boosting_final.joblib")

preprocessor = model.named_steps["preprocessor"]
regressor = model.named_steps["regressor"]

encoder = preprocessor.named_transformers_["categorical"]
scaler = preprocessor.named_transformers_["numerical"]

# Extract categorical feature categories
categories = {
    name: [str(value) for value in values]
    for name, values in zip(
        [
            "airline",
            "source_city",
            "departure_time",
            "stops",
            "arrival_time",
            "destination_city",
            "class",
        ],
        encoder.categories_,
    )
}

# Extract numerical scaling parameters
scaling = {
    "means": scaler.mean_.tolist(),
    "scales": scaler.scale_.tolist(),
}

# Extract bin thresholds used by HistGradientBoosting
bin_thresholds = [
    thresholds.tolist()
    for thresholds in regressor._bin_mapper.bin_thresholds_
]

# Export all 150 trees
trees = []

for iteration in regressor._predictors:
    tree = iteration[0]
    nodes = []

    for node in tree.nodes:
        nodes.append({
            "value": float(node["value"]),
            "feature_idx": int(node["feature_idx"]),
            "left": int(node["left"]),
            "right": int(node["right"]),
            "bin_threshold": int(node["bin_threshold"]),
            "is_leaf": bool(node["is_leaf"]),
            "missing_go_to_left": bool(node["missing_go_to_left"]),
        })

    trees.append(nodes)

# Final model data
exported_model = {
    "categories": categories,
    "scaling": scaling,
    "baseline": float(regressor._baseline_prediction[0, 0]),
    "learning_rate": float(regressor.learning_rate),
    "bin_thresholds": bin_thresholds,
    "trees": trees,
}

# Save inside docs for GitHub Pages
output_path = Path("docs/model.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as file:
    json.dump(exported_model, file, separators=(",", ":"))

print("Model exported successfully!")
print("Trees exported:", len(trees))
print("Output file:", output_path.resolve())
print("File size (MB):", round(output_path.stat().st_size / (1024 * 1024), 2))
