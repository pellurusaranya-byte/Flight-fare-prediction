"""
FOLLOW-UP to notebook 15: baselines inside each class, plus a global-median reference.

The proposal asked for results "reported separately for Economy and Business and always against the
baseline", and for "a global median as a reference". Notebook 15 reports the baselines overall; this script
adds the per-class view on the same final-test routes.

Same sample (100,000 rows, seed 42) and same final-test routes as notebook 15. Every baseline learns its
medians from the development routes only. Nothing here is used to change a model or a setting.
The model's per-class numbers come from results/final_test_predictions.csv (written by notebook 15).

Run from the project root (after notebook 15):  python notebooks/19_baseline_per_class.py
Output: results/baseline_per_class.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

df = pd.read_csv(ROOT / "data" / "Clean_Dataset.csv").drop(columns=["Unnamed: 0", "flight"])
df["route"] = ["-".join(sorted(p)) for p in zip(df["source_city"], df["destination_city"])]
df = df.sample(n=100_000, random_state=SEED).reset_index(drop=True)
dev_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=3 / 15, random_state=SEED)
                         .split(df, df["price"], df["route"]))
dev, test = df.iloc[dev_idx], df.iloc[test_idx]

global_med = float(dev["price"].median())
med_class = dev.groupby("class")["price"].median()
med_cs = dev.groupby(["class", "stops"])["price"].median()

preds = {
    "Reference: global median (all flights)": pd.Series(global_med, index=test.index),
    "Baseline: median price by class": test["class"].map(med_class),
    "Baseline: median price by class and stops": pd.Series(
        [med_cs.get(k, med_class[k[0]]) for k in zip(test["class"], test["stops"])], index=test.index),
}
model = pd.read_csv(OUT / "final_test_predictions.csv")     # same row order as the final-test split
assert len(model) == len(test)
preds["Gradient Boosting (final model)"] = pd.Series(model["predicted"].values, index=test.index)


def metrics(actual, pred):
    return {"MAE": float(mean_absolute_error(actual, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
            "R2": float(r2_score(actual, pred))}


rows = []
for name, p in preds.items():
    for cls in ["All", "Economy", "Business"]:
        m = np.ones(len(test), dtype=bool) if cls == "All" else (test["class"] == cls).values
        rows.append({"Model": name, "Class": cls, "Flights": int(m.sum()),
                     **metrics(test["price"][m], p[m])})
res = pd.DataFrame(rows)
res.to_csv(OUT / "baseline_per_class.csv", index=False)
print("global median (development routes):", global_med)
print(res.round(3).to_string(index=False))

# consistency checks against notebook 15's saved results
ft = pd.read_csv(OUT / "final_test_results.csv").set_index("Model")
chk = res[res["Class"] == "All"].set_index("Model")
for a, b in [("Baseline: median price by class", "Baseline: median price by class"),
             ("Baseline: median price by class and stops", "Baseline: median price by class and stops")]:
    assert abs(chk.loc[a, "MAE"] - ft.loc[b, "MAE"]) < 1e-6, a
gb = [m for m in ft.index if m.startswith("Gradient Boosting")][0]
assert abs(chk.loc["Gradient Boosting (final model)", "MAE"] - ft.loc[gb, "MAE"]) < 1e-6
print("\nchecks against final_test_results.csv: OK")
