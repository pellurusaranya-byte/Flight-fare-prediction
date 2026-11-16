# FlyWise – Flight Fare Prediction

 **Live demo:** https://pellurusaranya-byte.github.io/Flight-fare-prediction/  


**FlyWise** is a machine-learning-based web application that predicts domestic flight ticket prices in India based on flight search details. It helps users estimate airfare before booking their journey.

## Features

- Predict estimated flight fares using machine learning.
- Select source and destination cities.
- Choose airline, travel class, number of stops, departure time, and travel duration.
- Adjust the number of days remaining before departure.
- Simple and interactive web interface.
- Data analysis and model evaluation visualizations.

##  Technologies Used

- **Python** – Programming language
- **Flask** – Web application framework
- **Pandas and NumPy** – Data processing
- **Scikit-learn** – Machine learning
- **Joblib** – Saving and loading trained models
- **HTML, CSS, JavaScript** – Frontend development

##  Machine Learning Models

The project explores and compares three regression algorithms:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

Gradient Boosting was selected by cross-validation on 12 city pairs, then scored **once** on 3 city pairs that were never used for selection or tuning.

### Model Evaluation (final test: 3 unseen routes)

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Gradient Boosting (final model) | ₹3,609 | ₹6,079 | 0.911 |
| Random Forest | ₹3,931 | ₹6,556 | 0.897 |
| Linear Regression | ₹5,255 | ₹7,519 | 0.864 |
| Baseline: median by class and stops | ₹4,534 | ₹6,948 | 0.884 |
| Baseline: median by class | ₹5,491 | ₹8,856 | 0.812 |

*Final test = Bangalore–Chennai, Delhi–Hyderabad and Delhi–Mumbai (20,456 flights from a 100,000-row sample). MAE and RMSE are in Indian Rupees. Within Business class the model explains only about a quarter of the price variation (R² 0.25), so most of the headline R² comes from separating Economy from Business. See the project report for the full evaluation, error analysis and limitations.*

### Reproduce the results

```bash
pip install -r requirements.txt
python notebooks/15_final_evaluation.py   # CV on 12 routes, one-shot test on 3 routes, baselines, errors
python notebooks/16_error_figures.py      # report figures
python notebooks/12_train_final_model.py  # final model trained on all rows
python export_for_pages.py                # exports model + dropdown options to docs/data/
```

Notebooks 01 to 11 and 14 are earlier exploratory experiments (they use arrival time and a different split) and are superseded by notebook 15.

## Project Structure

```text
FlyWise/
│
├── data/
│   └── Clean_Dataset.csv
│
├── models/
│   ├── gradient_boosting_final.joblib
│   ├── random_forest_final.joblib
│   └── other trained models
│
├── notebooks/
│   └── model development and evaluation scripts (15 = final evaluation)
│
├── results/
│   └── evaluation graphs and result files
│
├── docs/                      (static site published on GitHub Pages)
│   ├── index.html, predict.html, report.html
│   ├── script.js, style.css
│   └── data/ (model.json, options.json)
│
├── website/                  
│   ├── app.py
│   ├── static/
│   │   ├── style.css
│   │   └── script.js
│   └── templates/
│       ├── index.html
│       └── predict.html
│
├── check_class_accuracy.py
├── .gitignore
└── README.md
```

##  Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/pellurusaranya-byte/Flight-fare-prediction.git
cd Flight-fare-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
cd website
python app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:5000/
```

## Dataset

The project uses a dataset containing approximately 300,153 domestic flight records.

The features include:

- Airline
- Source city
- Destination city
- Departure time
- Number of stops
- Flight duration
- Travel class
- Days left before departure

The target variable is the flight ticket price.

##  Disclaimer

FlyWise provides estimated flight fares based on historical data and machine learning predictions. These estimates are not live airline prices and may differ from actual booking prices.

##  Future Improvements

- Integrate live flight fare APIs.
- Deploy the application online.
- Improve prediction accuracy with additional data.
- Add automated testing and CI/CD workflows.

---

**Developed as an academic machine learning project.**
