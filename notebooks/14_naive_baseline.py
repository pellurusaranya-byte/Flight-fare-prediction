# Naive baseline: predict the median training price of each travel class.
# Uses the SAME route-held-out split as 10_robust_validation.py so numbers are comparable.
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "Clean_Dataset.csv").sample(n=50000, random_state=42).copy()
df["route_group"] = df.apply(lambda r: "_".join(sorted([r["source_city"], r["destination_city"]])), axis=1)
df = df.drop(columns=["Unnamed: 0", "flight"])
X, y, groups = df.drop(columns=["price", "route_group"]), df["price"], df["route_group"]

train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups=groups))
median_by_class = y.iloc[train].groupby(X["class"].iloc[train]).median()
pred = X["class"].iloc[test].map(median_by_class)
y_test = y.iloc[test]

row = {"Model": "Naive baseline (median price by class)",
       "MAE": round(mean_absolute_error(y_test, pred), 2),
       "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
       "R2 Score": round(r2_score(y_test, pred), 4)}
pd.DataFrame([row]).to_csv(ROOT / "results" / "naive_baseline.csv", index=False)
print(row)
