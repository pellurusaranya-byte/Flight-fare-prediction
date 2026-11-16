"""
Median absolute percentage error (MdAPE) on the final held-out routes.

The proposal promised MdAPE as a metric. It is computed here from the saved
final-test predictions (results/final_test_predictions.csv), so no model needs
to be retrained. Run from the project root:  python notebooks/17_mdape.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
pred = pd.read_csv(ROOT / "results" / "final_test_predictions.csv")

pred["ape_pct"] = pred["abs_error"] / pred["actual"] * 100

rows = [("All test flights", pred["ape_pct"])]
for cls, part in pred.groupby("class"):
    rows.append((cls, part["ape_pct"]))

out = pd.DataFrame(
    [{"group": name, "rows": len(s), "MdAPE_pct": round(s.median(), 1),
      "mean_APE_pct": round(s.mean(), 1)} for name, s in rows]
)
print(out.to_string(index=False))
out.to_csv(ROOT / "results" / "final_test_mdape.csv", index=False)
