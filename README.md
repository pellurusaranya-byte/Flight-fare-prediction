# ✈️ FlyWise – Flight Fare Prediction

**FlyWise** is a machine-learning-based web application that predicts domestic flight ticket prices in India based on flight search details. It helps users estimate airfare before booking their journey.

## 🌟 Features

- Predict estimated flight fares using machine learning.
- Select source and destination cities.
- Choose airline, travel class, number of stops, departure time, and travel duration.
- Adjust the number of days remaining before departure.
- Simple and interactive web interface.
- Data analysis and model evaluation visualizations.

## 🛠️ Technologies Used

- **Python** – Programming language
- **Flask** – Web application framework
- **Pandas and NumPy** – Data processing
- **Scikit-learn** – Machine learning
- **Joblib** – Saving and loading trained models
- **HTML, CSS, JavaScript** – Frontend development

## 🤖 Machine Learning Models

The project explores and compares three regression algorithms:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

Gradient Boosting was selected as the final model based on the model evaluation and validation results.

### Model Evaluation

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Gradient Boosting | ₹3,162.73 | ₹5,395.86 | 0.9411 |
| Random Forest | ₹3,341.73 | ₹5,698.75 | 0.9343 |
| Linear Regression | ₹4,687.60 | ₹6,952.41 | 0.9031 |

*These are GroupKFold validation results. MAE and RMSE are in Indian Rupees.*

## 📂 Project Structure

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
│   └── model development and evaluation scripts
│
├── results/
│   └── evaluation graphs and result files
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

## ⚙️ Installation and Setup

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
pip install flask pandas numpy scikit-learn joblib
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

## 📊 Dataset

The project uses a dataset containing approximately 300,153 domestic flight records.

The features include:

- Airline
- Source city
- Destination city
- Departure time
- Arrival time
- Number of stops
- Flight duration
- Travel class
- Days left before departure

The target variable is the flight ticket price.

## ⚠️ Disclaimer

FlyWise provides estimated flight fares based on historical data and machine learning predictions. These estimates are not live airline prices and may differ from actual booking prices.

## 🚀 Future Improvements

- Integrate live flight fare APIs.
- Deploy the application online.
- Improve prediction accuracy with additional data.
- Add automated testing and CI/CD workflows.

---

**Developed as an academic machine learning project.**
Link to website
https://pellurusaranya-byte.github.io/Flight-fare-prediction/
