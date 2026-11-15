"""
FINAL EVALUATION (replaces the evaluation logic of notebooks 10, 11 and 14).

Design (decided BEFORE looking at any score):
  1. Final test set = 3 whole city pairs (both directions together), chosen once with a
     fixed seed. It is NOT used for model selection or tuning.
  2. Development set = the other 12 city pairs. Models and settings are compared with
     4-fold GroupKFold CV (each fold hides 3 whole city pairs).
  3. The best setting by CV MAE is refit on the whole development set and scored ONCE
     on the final test routes. Baselines are scored on the same test routes.
  4. `arrival_time` is NOT a feature: a user searching for a fare picks departure time and
     duration, so arrival time is not known in the same way (and the website has no field for it).

Run from the project root:  python notebooks/15_final_evaluation.py
Outputs: results/*.csv and results/final_eval.json (used by the report page).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

CATEGORICAL = ["airline", "source_city", "departure_time", "stops",
               "destination_city", "class"]          # arrival_time deliberately left out
NUMERIC = ["duration", "days_left"]

# ------------------------------------------------------------------ data
df = pd.read_csv(ROOT / "data" / "Clean_Dataset.csv").drop(columns=["Unnamed: 0", "flight"])
df["route"] = ["-".join(sorted(p)) for p in zip(df["source_city"], df["destination_city"])]
# Work on a fixed 100,000-row sample so every script finishes in minutes on a laptop.
df = df.sample(n=100_000, random_state=SEED).reset_index(drop=True)
X, y, groups = df[CATEGORICAL + NUMERIC], df["price"], df["route"]


def metrics(actual, pred):
    return {"MAE": float(mean_absolute_error(actual, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
            "R2": float(r2_score(actual, pred))}


def prep(dense):
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense), CATEGORICAL),
        ("numerical", StandardScaler(), NUMERIC)])


def make(name):
    if name == "Linear Regression":
        return Pipeline([("p", prep(False)), ("m", LinearRegression())])
    if name == "Random Forest":
        return Pipeline([("p", prep(False)), ("m", RandomForestRegressor(
            n_estimators=30, max_depth=15, min_samples_leaf=5, random_state=SEED, n_jobs=-1))])
    # Gradient Boosting variants (a small tuning grid, judged by CV only)
    leaf, lr, it = {"Gradient Boosting (A: 31 leaves, lr 0.1, 150 it)": (31, 0.1, 150),
                    "Gradient Boosting (B: 15 leaves, lr 0.1, 150 it)": (15, 0.1, 150),
                    "Gradient Boosting (C: 31 leaves, lr 0.05, 300 it)": (31, 0.05, 300)}[name]
    return Pipeline([("p", prep(True)), ("m", HistGradientBoostingRegressor(
        max_leaf_nodes=leaf, learning_rate=lr, max_iter=it, l2_regularization=1.0, random_state=SEED))])


MODELS = ["Linear Regression", "Random Forest",
          "Gradient Boosting (A: 31 leaves, lr 0.1, 150 it)",
          "Gradient Boosting (B: 15 leaves, lr 0.1, 150 it)",
          "Gradient Boosting (C: 31 leaves, lr 0.05, 300 it)"]

# ------------------------------------------------------------------ 1. final test routes
final_split = GroupShuffleSplit(n_splits=1, test_size=3 / 15, random_state=SEED)
dev_idx, test_idx = next(final_split.split(X, y, groups))
X_dev, y_dev, g_dev = X.iloc[dev_idx], y.iloc[dev_idx], groups.iloc[dev_idx]
X_test, y_test, g_test = X.iloc[test_idx], y.iloc[test_idx], groups.iloc[test_idx]
test_routes = sorted(set(g_test))
print("FINAL TEST ROUTES (never used for selection):", test_routes)
print("dev rows", len(X_dev), "| test rows", len(X_test))

# ------------------------------------------------------------------ 2. CV on development routes
rows = []
for fold, (tr, va) in enumerate(GroupKFold(n_splits=4).split(X_dev, y_dev, g_dev), 1):
    held = sorted(set(g_dev.iloc[va]))
    for name in MODELS:
        t = time.time()
        m = make(name).fit(X_dev.iloc[tr], y_dev.iloc[tr])
        r = metrics(y_dev.iloc[va], m.predict(X_dev.iloc[va]))
        rows.append({"Fold": fold, "Model": name, "Held-out routes": ", ".join(held), **r})
        print(f"fold {fold} {name:55s} MAE {r['MAE']:8.1f}  R2 {r['R2']:.4f}  ({time.time()-t:.0f}s)")
    # baselines inside the same folds
    med = y_dev.iloc[tr].groupby(X_dev["class"].iloc[tr]).median()
    rows.append({"Fold": fold, "Model": "Baseline: median price by class", "Held-out routes": ", ".join(held),
                 **metrics(y_dev.iloc[va], X_dev["class"].iloc[va].map(med))})
cv = pd.DataFrame(rows)
cv.to_csv(OUT / "cv_fold_results.csv", index=False)
cv_sum = cv.groupby("Model").agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
                                 RMSE_mean=("RMSE", "mean"), R2_mean=("R2", "mean"),
                                 R2_std=("R2", "std")).reset_index().sort_values("MAE_mean")
cv_sum.to_csv(OUT / "cv_summary.csv", index=False)
print("\nCV SUMMARY (development routes only)\n", cv_sum.round(3).to_string(index=False))

# how many folds does the winner beat Random Forest?
cand = cv_sum[cv_sum["Model"].str.startswith("Gradient")].iloc[0]["Model"]
best_name = cand if cv_sum.set_index("Model").loc[cand, "MAE_mean"] < \
    cv_sum.set_index("Model").loc["Random Forest", "MAE_mean"] else "Random Forest"
piv = cv.pivot(index="Fold", columns="Model", values="MAE")
folds_won = int((piv[best_name] < piv["Random Forest"]).sum()) if best_name != "Random Forest" else None
print("SELECTED BY CV:", best_name, "| folds where it beats Random Forest:", folds_won, "/ 4")

# ------------------------------------------------------------------ 3. one-shot final test
final = {"selected": best_name, "test_routes": test_routes, "n_dev": int(len(X_dev)), "n_test": int(len(X_test)),
         "folds_won_vs_rf": folds_won}
fitted = {}
test_rows = []
for name in ["Linear Regression", "Random Forest", best_name]:
    if name in fitted:
        continue
    fitted[name] = make(name).fit(X_dev, y_dev)
    test_rows.append({"Model": name, **metrics(y_test, fitted[name].predict(X_test))})
med_class = y_dev.groupby(X_dev["class"]).median()
test_rows.append({"Model": "Baseline: median price by class",
                  **metrics(y_test, X_test["class"].map(med_class))})
med_cs = y_dev.groupby([X_dev["class"], X_dev["stops"]]).median()
key = list(zip(X_test["class"], X_test["stops"]))
test_rows.append({"Model": "Baseline: median price by class and stops",
                  **metrics(y_test, pd.Series([med_cs.get(k, med_class[k[0]]) for k in key], index=X_test.index))})
test = pd.DataFrame(test_rows)
test.to_csv(OUT / "final_test_results.csv", index=False)
print("\nFINAL TEST (3 unseen routes, scored once)\n", test.round(3).to_string(index=False))
final["test"] = test.round(4).to_dict("records")

# ------------------------------------------------------------------ 4. error analysis of the selected model
pred = fitted[best_name].predict(X_test)
err = pd.DataFrame({"route": g_test.values, "airline": X_test["airline"].values, "class": X_test["class"].values,
                    "stops": X_test["stops"].values, "departure_time": X_test["departure_time"].values,
                    "duration": X_test["duration"].values, "days_left": X_test["days_left"].values,
                    "actual": y_test.values, "predicted": pred})
err["error"] = err["actual"] - err["predicted"]
err["abs_error"] = err["error"].abs()
err["pct_error"] = err["abs_error"] / err["actual"] * 100
err.to_csv(OUT / "final_test_predictions.csv", index=False)
err.sort_values("abs_error", ascending=False).head(10).round(1).to_csv(OUT / "worst_predictions.csv", index=False)
print("\nWORST 10\n", err.sort_values("abs_error", ascending=False).head(10).round(0).to_string(index=False))

per_class = [{"Class": c, "Flights": int(len(d)), **metrics(d["actual"], d["predicted"]),
              "MAPE": float(d["pct_error"].mean())} for c, d in err.groupby("class")]
pd.DataFrame(per_class).to_csv(OUT / "final_test_per_class.csv", index=False)
final["per_class"] = per_class
per_route = [{"Route": r, "Flights": int(len(d)), **metrics(d["actual"], d["predicted"])} for r, d in err.groupby("route")]
pd.DataFrame(per_route).to_csv(OUT / "final_test_per_route.csv", index=False)
final["per_route"] = per_route
print("\nPER CLASS", per_class, "\nPER ROUTE", per_route)

# error by price bucket
bins = [0, 5000, 10000, 20000, 40000, 1e9]
labels = ["Under ₹5,000", "₹5,000 to 10,000", "₹10,000 to 20,000", "₹20,000 to 40,000", "Over ₹40,000"]
err["bucket"] = pd.cut(err["actual"], bins=bins, labels=labels, right=False)
buckets = err.groupby("bucket", observed=True).agg(Flights=("abs_error", "size"), MAE=("abs_error", "mean")).reset_index()
final["buckets"] = [{"bucket": str(r.bucket), "flights": int(r.Flights), "mae": float(r.MAE)} for r in buckets.itertuples()]
final["median_abs_error"] = float(err["abs_error"].median())
# cumulative share within tolerance (steps of 250 rupees, 0..10000)
final["cdf"] = [round(float((err["abs_error"] <= t).mean() * 100), 1) for t in range(0, 10001, 250)]
# worst cases for the report table
final["worst"] = err.sort_values("abs_error", ascending=False).head(8).round(0).to_dict("records")

# ------------------------------------------------------------------ 5. contrast: easy random split (leaky on purpose)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
rand_rows = []
for name in ["Linear Regression", "Random Forest", best_name]:
    if name in [r["Model"] for r in rand_rows]:
        continue
    rand_rows.append({"Model": name, **metrics(yte, make(name).fit(Xtr, ytr).predict(Xte))})
mc = ytr.groupby(Xtr["class"]).median()
rand_rows.append({"Model": "Baseline: median price by class", **metrics(yte, Xte["class"].map(mc))})
rand = pd.DataFrame(rand_rows)
rand.to_csv(OUT / "random_split_results.csv", index=False)
final["random_split"] = rand.round(4).to_dict("records")
final["cv"] = cv_sum.round(4).to_dict("records")
print("\nRANDOM SPLIT (easy, same routes in train and test)\n", rand.round(3).to_string(index=False))

(OUT / "final_eval.json").write_text(json.dumps(final, indent=1, default=str))
print("\nsaved results/final_eval.json")
