"""
Exports the trained model + dropdown options to JSON so the website can run
entirely in the browser (GitHub Pages cannot run Python/Flask).
Run from the project root:  python export_for_pages.py
"""
import json
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
model = joblib.load(ROOT / "models" / "gradient_boosting_final.joblib")
pre = model.named_steps["preprocessor"]
reg = model.named_steps["regressor"]

cat_cols = ["airline", "source_city", "departure_time", "stops",
            "destination_city", "class"]          # arrival_time is not a feature any more
categories = {c: list(cats) for c, cats in
              zip(cat_cols, pre.named_transformers_["categorical"].categories_)}
scaler = pre.named_transformers_["numerical"]

trees = []
for iteration in reg._predictors:
    p = iteration[0]
    n = p.nodes
    trees.append({
        "f": n["feature_idx"].tolist(),
        "t": [float(x) for x in n["num_threshold"]],
        "l": n["left"].tolist(),
        "r": n["right"].tolist(),
        "v": [round(float(x), 3) for x in n["value"]],
        "leaf": n["is_leaf"].astype(int).tolist(),
        "ml": n["missing_go_to_left"].astype(int).tolist(),
    })

model_json = {
    "categories": categories,
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "baseline": float(reg._baseline_prediction.ravel()[0]),
    "trees": trees,
}
(ROOT / "docs/data/model.json").write_text(json.dumps(model_json, separators=(",", ":")))

# Dropdown options: source -> destination -> stops -> airline -> class -> departure -> [durations]
# Class sits inside the chain so a visitor can only pick combinations that exist in the data
# (e.g. Business exists only for Air_India and Vistara, and not on every route).
df = pd.read_csv(ROOT / "data" / "Clean_Dataset.csv")
df["dur"] = (df["duration"] * 2).round() / 2
options = {}
g = df.groupby(["source_city", "destination_city", "stops", "airline", "class", "departure_time"])["dur"]
for (s, d, st, a, c, dep), vals in g:
    options.setdefault(s, {}).setdefault(d, {}).setdefault(st, {}) \
           .setdefault(a, {}).setdefault(c, {})[dep] = sorted(set(float(v) for v in vals))
(ROOT / "docs/data/options.json").write_text(json.dumps(options, separators=(",", ":")))
print("done")
