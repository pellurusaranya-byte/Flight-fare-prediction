"""
FOLLOW-UP to notebook 15: the two proposal items that were planned but not run in the main comparison.
  (a) Ridge regression (alpha chosen by the same 4-fold route CV, development routes only)
  (b) Modelling log(price) instead of price in rupees (Linear, Ridge and Gradient Boosting)

Same sample (100,000 rows, seed 42), same final-test routes, same GroupKFold as notebook 15.
Selection (which alpha) uses development-route CV only. The final-test routes are scored once per
model afterwards for reporting; nothing here is used to change the selected model.
MAE/RMSE/R2 are always computed on the price in rupees (log predictions are back-transformed with exp).

Run from the project root:  python notebooks/18_ridge_and_log_target.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
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

def metrics(a, p):
    return {"MAE": float(mean_absolute_error(a, p)), "RMSE": float(np.sqrt(mean_squared_error(a, p))),
            "R2": float(r2_score(a, p))}

def prep(dense=False):
    return ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense), CATEGORICAL),
                              ("n", StandardScaler(), NUMERIC)])

def gb():
    return HistGradientBoostingRegressor(max_leaf_nodes=15, learning_rate=0.1, max_iter=150,
                                         l2_regularization=1.0, random_state=SEED)

def logged(reg):
    return TransformedTargetRegressor(regressor=reg, func=np.log, inverse_func=np.exp)

ALPHAS = [0.1, 1, 10, 100, 1000]
MODELS = {"Linear Regression (₹)": lambda: Pipeline([("p", prep()), ("m", LinearRegression())])}
for a in ALPHAS:
    MODELS[f"Ridge a={a} (₹)"] = (lambda a=a: Pipeline([("p", prep()), ("m", Ridge(alpha=a))]))
MODELS["Linear Regression (log price)"] = lambda: Pipeline([("p", prep()), ("m", logged(LinearRegression()))])
for a in ALPHAS:
    MODELS[f"Ridge a={a} (log price)"] = (lambda a=a: Pipeline([("p", prep()), ("m", logged(Ridge(alpha=a)))]))
MODELS["Gradient Boosting B (₹)"] = lambda: Pipeline([("p", prep(True)), ("m", gb())])
MODELS["Gradient Boosting B (log price)"] = lambda: Pipeline([("p", prep(True)), ("m", logged(gb()))])

split = GroupShuffleSplit(n_splits=1, test_size=3 / 15, random_state=SEED)
dev, te = next(split.split(X, y, groups))
Xd, yd, gd, Xt, yt = X.iloc[dev], y.iloc[dev], groups.iloc[dev], X.iloc[te], y.iloc[te]
print("test routes:", sorted(set(groups.iloc[te])))

rows = []
for fold, (tr, va) in enumerate(GroupKFold(n_splits=4).split(Xd, yd, gd), 1):
    for name, mk in MODELS.items():
        m = mk().fit(Xd.iloc[tr], yd.iloc[tr])
        rows.append({"Fold": fold, "Model": name, **metrics(yd.iloc[va], m.predict(Xd.iloc[va]))})
cv = pd.DataFrame(rows)
s = cv.groupby("Model").agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
                            R2_mean=("R2", "mean")).reset_index()
trows = []
for name, mk in MODELS.items():
    trows.append({"Model": name, **metrics(yt, mk().fit(Xd, yd).predict(Xt))})
t = pd.DataFrame(trows).rename(columns={"MAE": "Test_MAE", "RMSE": "Test_RMSE", "R2": "Test_R2"})
out = s.merge(t, on="Model")
out["order"] = out["Model"].map({k: i for i, k in enumerate(MODELS)})
out = out.sort_values("order").drop(columns="order")
out.to_csv(OUT / "ridge_log_target_results.csv", index=False)
cv.to_csv(OUT / "ridge_log_target_fold_results.csv", index=False)
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))

# per-class view of the log effect on the final test (does it help Economy?)
for label, mk in [("GB raw", MODELS["Gradient Boosting B (₹)"]), ("GB log", MODELS["Gradient Boosting B (log price)"]),
                  ("Linear raw", MODELS["Linear Regression (₹)"]), ("Linear log", MODELS["Linear Regression (log price)"])]:
    p = mk().fit(Xd, yd).predict(Xt)
    for c in ["Economy", "Business"]:
        k = (Xt["class"] == c).values
        print(f"{label:11s} {c:9s} MAE {mean_absolute_error(yt[k], p[k]):8.0f}  "
              f"MdAPE {np.median(np.abs(yt[k]-p[k])/yt[k])*100:5.1f}%  bias(pred-actual) {np.mean(p[k]-yt[k].values):8.0f}")
