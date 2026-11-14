
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from pathlib import Path


# 1. Find the main project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 2. Create Flask application
app = Flask(__name__)


# 3. Load the trained ML model
MODEL_PATH = PROJECT_ROOT / "models" / "gradient_boosting_final.joblib"

model = joblib.load(MODEL_PATH)

print("FlyWise ML model loaded successfully!")

# Load the original flight dataset
DATA_PATH = PROJECT_ROOT / "data" / "Clean_Dataset.csv"
flight_dataset = pd.read_csv(DATA_PATH)

print("Flight dataset loaded successfully!")

# 4. Home page
@app.route("/")
def home():
    return render_template("index.html", prediction=None)


@app.route("/get-airlines")
def get_airlines():

    source = request.args.get("source", "").strip()
    destination = request.args.get("destination", "").strip()
    stops = request.args.get("stops", "").strip()

    if not source or not destination or not stops:
        return jsonify([])

    # Convert common display labels into dataset values
    stop_mapping = {
        "non-stop": "zero",
        "non stop": "zero",
        "0": "zero",
        "0 stops": "zero",
        "1 stop": "one",
        "one stop": "one",
        "2+ stops": "two_or_more",
        "2 or more stops": "two_or_more"
    }

    stops = stop_mapping.get(stops.lower(), stops)

    matching_flights = flight_dataset[
        (flight_dataset["source_city"] == source) &
        (flight_dataset["destination_city"] == destination) &
        (flight_dataset["stops"] == stops)
    ]

    airlines = sorted(
        matching_flights["airline"].dropna().unique().tolist()
    )

    return jsonify(airlines)


@app.route("/get-departure-times")
def get_departure_times():

    source = request.args.get("source")
    destination = request.args.get("destination")
    stops = request.args.get("stops")
    airline = request.args.get("airline")

    if not all([source, destination, stops, airline]):
        return jsonify([])

    matching_flights = flight_dataset[
        (flight_dataset["source_city"] == source) &
        (flight_dataset["destination_city"] == destination) &
        (flight_dataset["stops"] == stops) &
        (flight_dataset["airline"] == airline)
    ]

    departure_times = sorted(
        matching_flights["departure_time"].unique().tolist()
    )

    return jsonify(departure_times)
  
@app.route("/get-durations")
def get_durations():

    source = request.args.get("source")
    destination = request.args.get("destination")
    stops = request.args.get("stops")
    airline = request.args.get("airline")
    departure_time = request.args.get("departure_time")

    if not all([source, destination, stops, airline, departure_time]):
        return jsonify([])

    matching_flights = flight_dataset[
        (flight_dataset["source_city"] == source) &
        (flight_dataset["destination_city"] == destination) &
        (flight_dataset["stops"] == stops) &
        (flight_dataset["airline"] == airline) &
        (flight_dataset["departure_time"] == departure_time)
    ]

    durations = (
        (matching_flights["duration"] * 2).round() / 2
    ).unique()

    durations = sorted(durations)

    return jsonify([float(d) for d in durations])

@app.route("/predict-page")
def predict_page():
    return render_template("predict.html", prediction=None)

# 5. Handle flight price prediction

@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get information submitted by the user
        airline = request.form["airline"]
        source_city = request.form["source_city"]
        destination_city = request.form["destination_city"]
        departure_time = request.form["departure_time"]
        stops = request.form["stops"]
        travel_class = request.form["class"]

        duration = float(request.form["duration"])
        days_left = int(request.form["days_left"])

        # Temporarily estimate arrival category
        # We will improve this calculation later.
        arrival_time = departure_time

        # Prepare input in the same format used during training
        flight_data = pd.DataFrame([{
            "airline": airline,
            "source_city": source_city,
            "departure_time": departure_time,
            "stops": stops,
            "arrival_time": arrival_time,
            "destination_city": destination_city,
            "class": travel_class,
            "duration": duration,
            "days_left": days_left
        }])

        # Predict flight price
        predicted_price = model.predict(flight_data)[0]
        predicted_price = round(float(predicted_price), 2)

        # Display prediction on the prediction page
        return render_template(
            "predict.html",
            prediction=predicted_price,
            flight_details=request.form
        )

    except (ValueError, KeyError) as error:

        print("Prediction error:", error)

        return render_template(
            "predict.html",
            error="Please enter valid flight details.",
            flight_details=request.form
        )



# 6. Run the application
if __name__ == "__main__":
    app.run(debug=True)
