# FlyWise – Predicting Indian Domestic Flight Prices

ES 335 course project (Machine Learning), IIT Gandhinagar. Author: Saranya Pelluru (25110237).

- **Website and project report:** https://pellurusaranya-byte.github.io/Flight-fare-prediction/
- **Repository:** https://github.com/pellurusaranya-byte/Flight-fare-prediction

The website is the main write-up: it explains the question, data, preprocessing, models, evaluation, results, errors and limitations, with interactive charts. This repository holds the code, data and result files needed to check and reproduce everything on it.

## Research question

How well can the price of a domestic Indian flight ticket be estimated from information a traveller sees when searching (airline, route, class, stops, departure time, duration and days left before departure), and which of these factors explain most of the variation in price? We also test whether the reported accuracy still holds when flights are kept out of training.

One row of the data is one fare listing for a specific flight and class on a route, observed a given number of days before departure. The claim is about estimation and association only, not about causes (for example, not that booking earlier lowers a fare).

## Main result

Models were compared on 3 city pairs that were never used for selection or tuning (Bangalore–Chennai, Delhi–Hyderabad, Delhi–Mumbai; 20,456 listings). Errors are in Indian rupees.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Gradient Boosting (final model) | ₹3,609 | ₹6,079 | 0.911 |
| Random Forest (best of three settings) | ₹3,931 | ₹6,556 | 0.897 |
| Linear Regression | ₹5,255 | ₹7,519 | 0.864 |
| Baseline: median by class and stops | ₹4,534 | ₹6,948 | 0.884 |
| Baseline: median by class | ₹5,491 | ₹8,856 | 0.812 |

Median absolute percentage error of the final model on the same routes is 17.8% (Economy 19.3%, Business 15.7%). Inside Business class the model explains only about a quarter of the price variation (R² 0.25), so most of the headline R² comes from separating Economy from Business. The report has the full error analysis and limitations.

## Data access

- **Source:** Kaggle dataset "Flight Price Prediction" by Shubham Bathwal, scraped from the EaseMyTrip booking website (https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction). It is free to download with a Kaggle account.
- **Included copy:** `data/Clean_Dataset.csv` (300,153 rows, 1,561 distinct flight codes, 6 metro cities, collected 11 Feb to 31 Mar 2022). To fetch it again, download the dataset from Kaggle and save `Clean_Dataset.csv` at that path.
- **Columns used:** airline, source city, destination city, departure time slot, stops, duration, class, days left, and price (the target). The index column and the flight code are dropped; the arrival time slot is not used because a traveller does not enter it when searching (see "Changes from the proposal").

## Setup and reproduction

Requires Python 3.11 or newer. Library versions are pinned in `requirements.txt`.

```bash
git clone https://github.com/pellurusaranya-byte/Flight-fare-prediction.git
cd Flight-fare-prediction
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python notebooks/15_final_evaluation.py   # cross-validation, final test, flight-code split, baselines, errors (about 30 min on one CPU core)
python notebooks/16_error_figures.py      # report figures for the held-out routes
python notebooks/17_mdape.py              # median absolute percentage error on the final test
python notebooks/18_ridge_and_log_target.py  # follow-up: Ridge and log(price) target
python notebooks/19_baseline_per_class.py    # baselines inside each class and the global-median reference
python notebooks/12_train_final_model.py  # final model trained on all rows
python export_for_pages.py                # exports the model and dropdown options to docs/data/
python export_explorer_data.py            # builds docs/data/explorer.json for the Explore fares page
```

Outputs are written to `results/` (for example `final_test_results.csv`, `cv_summary.csv`, `flight_grouped_split_results.csv`, `final_test_per_class.csv`, `final_test_mdape.csv`, `baseline_per_class.csv`, `final_test_predictions.csv`).

### Exact preprocessing and splits

Everything below is in `notebooks/15_final_evaluation.py` and uses random seed 42.

1. Take a random sample of 100,000 rows (`df.sample(n=100_000, random_state=42)`).
2. Define the route of a listing as the sorted pair of its source and destination city (15 routes).
3. **Final test:** `GroupShuffleSplit(test_size=3/15, random_state=42)` on the route. This holds out 3 whole routes (Bangalore–Chennai, Delhi–Hyderabad, Delhi–Mumbai). They are never used for selection or tuning.
4. **Model selection:** `GroupKFold(n_splits=4)` over the remaining 12 routes, where each fold hides 3 whole routes. Three Gradient Boosting settings and three Random Forest settings are compared this way, and the lowest mean MAE wins in each family.
5. **Preprocessing** is inside a scikit-learn pipeline and fitted on training data only: one-hot encoding of the categorical columns and standard scaling of the numeric columns (duration, days left).
6. For comparison, the same models are also scored on a naive random 80/20 split of rows, to show how much that inflates the scores.
7. Also for comparison, a flight-code grouped 80/20 split (`GroupShuffleSplit(test_size=0.2, random_state=42)` on the flight code, 1,207 codes in training and 302 in test) keeps every listing of a flight on one side. It is not used for selection. Random Forest MAE is ₹1,895 on the random split, ₹2,807 on this split and ₹3,931 on the final test routes.

The deployed model (`models/gradient_boosting_final.joblib`, trained by notebook 12) uses the chosen settings on all 300,153 rows, so it has no separate held-out score; the final-test table above is the estimate for that recipe.

Notebooks 01 to 11 and 14 are earlier exploratory experiments (some use arrival time and a different split). They are kept for transparency and are superseded by notebook 15. Their outputs are in `results/exploratory/`, and their numbers (for example the baseline on a 50,000-row sample) differ from the report. The report figures are the notebook 01 plots (`price_distribution.png`, `class_prices.png`, in `results/exploratory/`) and the notebook 16 plots (in `results/`); `docs/images/` holds copies.

## Changes from the proposal

The submitted proposal planned some things differently. The full table with reasons is in section 10 of the report. In short:

- The final test holds out whole city pairs instead of whole flight codes. A flight-code grouped 80/20 split was run only as an extra contrast (`results/flight_grouped_split_results.csv`), not for selection or tuning.
- Baselines are the median by class, and the median by class and stops, because a route-level median does not exist for unseen routes.
- Plain Linear Regression is used in the main comparison. Ridge and a log(price) target were checked afterwards (`notebooks/18_ridge_and_log_target.py`, results in `results/ridge_log_target_results.csv`): neither changes the conclusions.
- MdAPE was computed afterwards from the saved predictions (`notebooks/17_mdape.py`).
- Baselines inside each class and a global-median reference were added afterwards (`notebooks/19_baseline_per_class.py`, results in `results/baseline_per_class.csv`). The model beats the class-and-stops baseline by about 41% in Economy but only about 10% in Business.
- The arrival time slot is not used. Experiments use a 100,000-row sample; the deployed model uses all rows.
- The interactive "model curve vs baseline over 1 to 49 days left" explorer was left out on purpose: both baselines ignore days left (a flat line), and a price-vs-days-left curve invites booking-time advice the data cannot support. The flight-grouped toggle was replaced by the stricter route split. The site has a fare predictor (with a days-left input), a historical fare explorer, a split and metric toggle, and an error-tolerance slider instead.

## Repository structure

```text
data/                  Clean_Dataset.csv
models/                trained models (gradient_boosting_final.joblib is the deployed one)
notebooks/             numbered scripts; 15 = final evaluation, 12 = final model
results/               result tables, figures and the final-test predictions (notebooks 15 to 18)
  exploratory/         outputs of the superseded notebooks 01 to 11 and 14
docs/                  static website published on GitHub Pages
  index.html           project report (landing page)
  predict.html         fare predictor (runs the exported model in the browser)
  explore.html         historical fare explorer
  data/                model.json, options.json, explorer.json
website/               original Flask version of the predictor (optional)
export_for_pages.py    exports the model to docs/data/
export_explorer_data.py  builds docs/data/explorer.json
check_class_accuracy.py   helper script
requirements.txt
```

## AI-use disclosure

I used Python, pandas, NumPy, scikit-learn and Matplotlib for the analysis, VS code, and GitHub Pages to host this site. I used an AI assistant (Claude) for planning the project, writing and debugging some of the code (the evaluation scripts, error analysis, the Ridge and log-target follow-up, and the JSON export for the web page), for editing the report text, and for guidance and code while building this web page. I ran and checked the code and outputs myself, and I am responsible for every result. The data is the public CC0 Kaggle “Flight Price Prediction” dataset by Shubham Bathwal. It contains no personal information, and the repository contains no passwords or API keys.

## References

- Bathwal, S. *Flight Price Prediction*, Kaggle dataset (CC0): https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research* 12, 2825–2830.
- pandas, NumPy, Matplotlib and seaborn for data handling and figures; Chart.js for the Explore fares charts.

## Note

FlyWise gives estimates based on a historical snapshot of listed prices from one booking site (Feb–Mar 2022). They are not live airline prices and may differ from what travellers actually pay.
