"""
Figures for the report, all computed on the FINAL TEST ROUTES (cities pairs never used for
selection or tuning). Re-creates the same split as 15_final_evaluation.py (same seed), refits the
selected Gradient Boosting setting on the development routes, then plots its behaviour on the test routes.

Run from the project root:  python notebooks/16_error_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
CATEGORICAL = ["airline", "source_city", "departure_time", "stops", "destination_city", "class"]
NUMERIC = ["duration", "days_left"]

df = pd.read_csv(ROOT / "data" / "Clean_Dataset.csv").drop(columns=["Unnamed: 0", "flight"])
df["route"] = ["-".join(sorted(p)) for p in zip(df["source_city"], df["destination_city"])]
df = df.sample(n=100_000, random_state=SEED).reset_index(drop=True)
X, y, groups = df[CATEGORICAL + NUMERIC], df["price"], df["route"]
dev, test = next(GroupShuffleSplit(n_splits=1, test_size=3 / 15, random_state=SEED).split(X, y, groups))
print("test routes:", sorted(set(groups.iloc[test])))

model = Pipeline([
    ("p", ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
                             ("n", StandardScaler(), NUMERIC)])),
    ("m", HistGradientBoostingRegressor(max_leaf_nodes=15, learning_rate=0.1, max_iter=150,
                                        l2_regularization=1.0, random_state=SEED))])
model.fit(X.iloc[dev], y.iloc[dev])
pred = model.predict(X.iloc[test])
actual = y.iloc[test].values
cls = X.iloc[test]["class"].values
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# 1. actual vs predicted
fig, ax = plt.subplots(figsize=(6.4, 5.6))
for c, colr in [("Economy", "#2563eb"), ("Business", "#f59e0b")]:
    m = cls == c
    ax.scatter(actual[m], pred[m], s=4, alpha=0.25, color=colr, label=c)
lim = max(actual.max(), pred.max()) * 1.02
ax.plot([0, lim], [0, lim], "--", color="#334155", lw=1)
ax.set_xlabel("Actual price (₹)"); ax.set_ylabel("Predicted price (₹)")
ax.set_title("Held-out routes: actual vs predicted"); ax.legend(markerscale=3)
fig.tight_layout(); fig.savefig(OUT / "actual_vs_predicted.png", dpi=130); plt.close(fig)

# 2. errors by class
rows = []
for c in ["Economy", "Business"]:
    m = cls == c
    ae = np.abs(actual[m] - pred[m])
    rows.append((c, ae.mean(), (ae / actual[m]).mean() * 100))
fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
axes[0].bar([r[0] for r in rows], [r[1] for r in rows], color=["#2563eb", "#f59e0b"])
axes[0].set_title("Average miss (MAE, ₹)")
axes[1].bar([r[0] for r in rows], [r[2] for r in rows], color=["#2563eb", "#f59e0b"])
axes[1].set_title("Average miss (% of actual price)")
for a, k in zip(axes, [1, 2]):
    for i, r in enumerate(rows):
        a.text(i, r[k], f"{r[k]:,.0f}" if k == 1 else f"{r[k]:.1f}%", ha="center", va="bottom")
fig.tight_layout(); fig.savefig(OUT / "class_wise_errors.png", dpi=130); plt.close(fig)

# 3. permutation importance on the held-out routes (increase in MAE when a column is shuffled)
rng = np.random.RandomState(SEED)
sub = rng.choice(len(test), size=6000, replace=False)
imp = permutation_importance(model, X.iloc[test].iloc[sub], y.iloc[test].iloc[sub],
                             scoring="neg_mean_absolute_error", n_repeats=5, random_state=SEED)
pi = pd.DataFrame({"Feature": X.columns, "MAE increase (₹)": imp.importances_mean,
                   "std": imp.importances_std}).sort_values("MAE increase (₹)", ascending=False)
pi.to_csv(OUT / "permutation_importance.csv", index=False)
print(pi.round(1).to_string(index=False))
fig, ax = plt.subplots(figsize=(7, 4.2))
p = pi.iloc[::-1]
ax.barh(p["Feature"], p["MAE increase (₹)"], xerr=p["std"], color="#2563eb")
ax.set_xlabel("Increase in MAE when the column is shuffled (₹)")
ax.set_title("Permutation importance on held-out routes")
fig.tight_layout(); fig.savefig(OUT / "feature_importance.png", dpi=130); plt.close(fig)
print("figures saved")
