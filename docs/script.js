// FlyWise – runs completely in the browser (no Flask / server needed).
// It loads two JSON files: options.json (dropdown choices that exist in the data) and model.json (the trained model).

document.addEventListener("DOMContentLoaded", async function () {

    const source = document.querySelector("#source_city");
    const destination = document.querySelector("#destination_city");
    const stops = document.querySelector("#stops");
    const airline = document.querySelector("#airline");
    const departure = document.querySelector("#departure_time");
    const duration = document.querySelector("#duration");
    const travelClass = document.querySelector("#travel_class");
    const daysLeftInput = document.querySelector('input[name="days_left"]');
    const flightForm = document.querySelector(".flight-form");
    const predictButton = document.querySelector(".predict-button");
    const resultBox = document.querySelector("#result");
    const resultPrice = document.querySelector("#result-price");
    const errorBox = document.querySelector("#error");

    // ---------- 1. LOAD DATA ----------
    let options, model;
    try {
        const [o, m] = await Promise.all([
            fetch("data/options.json").then(r => r.json()),
            fetch("data/model.json").then(r => r.json())
        ]);
        options = o; model = m;
    } catch (e) {
        errorBox.textContent = "Could not load model files. Open this site through GitHub Pages (or a local server), not by double-clicking the file.";
        errorBox.hidden = false;
        console.error(e);
        return;
    }

    // ---------- 2. DROPDOWN HELPERS ----------
    const timeLabels = {
        "Early_Morning": "Early Morning (5 AM – 8 AM)",
        "Morning": "Morning (8 AM – 12 PM)",
        "Afternoon": "Afternoon (12 PM – 4 PM)",
        "Evening": "Evening (4 PM – 8 PM)",
        "Night": "Night (8 PM – 12 AM)",
        "Late_Night": "Late Night (12 AM – 5 AM)"
    };
    const timeOrder = Object.keys(timeLabels);

    function resetDropdown(dropdown, message) {
        dropdown.innerHTML = "";
        const option = document.createElement("option");
        option.value = "";
        option.textContent = message;
        dropdown.appendChild(option);
        dropdown.disabled = true;
    }

    function fill(dropdown, placeholder, values, labelFn) {
        resetDropdown(dropdown, values.length ? placeholder : "Not available");
        values.forEach(v => {
            const option = document.createElement("option");
            option.value = v;
            option.textContent = labelFn ? labelFn(v) : v;
            dropdown.appendChild(option);
        });
        dropdown.disabled = values.length === 0;
    }

    // ---------- 3. CASCADING DROPDOWNS (replaces the old Flask /get-... routes) ----------
    function updateStops() {
        stops.value = "";
        resetDropdown(airline, "Select stops first");
        resetDropdown(travelClass, "Select airline first");
        resetDropdown(departure, "Select travel class first");
        resetDropdown(duration, "Select departure time first");

        if (!source.value || !destination.value) { stops.disabled = true; return; }
        if (source.value === destination.value) {
            destination.setCustomValidity("Source and destination cannot be the same.");
            destination.reportValidity();
            stops.disabled = true;
            return;
        }
        destination.setCustomValidity("");
        stops.disabled = false;
    }

    // Each step below reads only the part of options.json that matches the earlier choices, so a
    // visitor can only pick combinations that exist in the data (for example Business class
    // exists only for Air India and Vistara, and not on every route).
    const pick = (...keys) => keys.reduce((node, k) => (node || {})[k], options) || {};

    function updateAirlines() {
        resetDropdown(travelClass, "Select airline first");
        resetDropdown(departure, "Select travel class first");
        resetDropdown(duration, "Select departure time first");
        const airlines = Object.keys(pick(source.value, destination.value, stops.value)).sort();
        fill(airline, "Select an airline", airlines, n => n.replaceAll("_", " "));
        if (!airlines.length && stops.value) resetDropdown(airline, "No airlines available");
    }

    function updateClasses() {
        resetDropdown(departure, "Select travel class first");
        resetDropdown(duration, "Select departure time first");
        const classes = Object.keys(pick(source.value, destination.value, stops.value, airline.value)).sort();
        fill(travelClass, "Select travel class", classes);
    }

    function updateDepartureTimes() {
        resetDropdown(duration, "Select departure time first");
        const times = Object.keys(pick(source.value, destination.value, stops.value, airline.value, travelClass.value))
            .sort((a, b) => timeOrder.indexOf(a) - timeOrder.indexOf(b));
        fill(departure, "Select departure time", times, t => timeLabels[t] || t);
    }

    function updateDurations() {
        const hours = pick(source.value, destination.value, stops.value, airline.value, travelClass.value, departure.value);
        fill(duration, "Select flight duration", Array.isArray(hours) ? hours : [], h => `${h} hours`);
    }

    source.addEventListener("change", updateStops);
    destination.addEventListener("change", updateStops);
    stops.addEventListener("change", updateAirlines);
    airline.addEventListener("change", updateClasses);
    travelClass.addEventListener("change", updateDepartureTimes);
    departure.addEventListener("change", updateDurations);

    daysLeftInput.addEventListener("input", function () {
        const days = Number(daysLeftInput.value);
        daysLeftInput.setCustomValidity(
            days < 1 || days > 49 ? "Days before departure must be between 1 and 49." : ""
        );
    });

    // ---------- 4. THE MODEL (replaces model.predict in app.py) ----------
    // Step A: turn the form into the same 31 numbers the model saw in training
    //         (one-hot columns for each category, then scaled duration & days_left).
    function buildFeatures(f) {
        const x = [];
        Object.keys(model.categories).forEach(col => {
            model.categories[col].forEach(cat => x.push(f[col] === cat ? 1 : 0));
        });
        x.push((f.duration - model.mean[0]) / model.scale[0]);
        x.push((f.days_left - model.mean[1]) / model.scale[1]);
        return x;
    }

    // Step B: walk every decision tree down to a leaf, then add up all leaf values.
    function predictPrice(f) {
        const x = buildFeatures(f);
        let total = model.baseline;
        for (const tree of model.trees) {
            let node = 0;
            while (!tree.leaf[node]) {
                node = x[tree.f[node]] <= tree.t[node] ? tree.l[node] : tree.r[node];
            }
            total += tree.v[node];
        }
        return total;
    }

    // ---------- 5. FORM SUBMIT ----------
    flightForm.addEventListener("submit", function (event) {
        event.preventDefault();
        errorBox.hidden = true;

        if (source.value === destination.value) {
            destination.setCustomValidity("Source and destination cannot be the same.");
            destination.reportValidity();
            return;
        }
        destination.setCustomValidity("");
        if (!flightForm.checkValidity()) { flightForm.reportValidity(); return; }

        // Safety net: only predict for a flight combination that exists in the data.
        const dur = parseFloat(duration.value);
        const known = pick(source.value, destination.value, stops.value, airline.value, travelClass.value, departure.value);
        if (!Array.isArray(known) || !known.includes(dur)) {
            errorBox.textContent = "That combination of airline, class, time and duration does not exist in our data, so no fare is estimated.";
            errorBox.hidden = false;
            return;
        }

        const flight = {
            airline: airline.value,
            source_city: source.value,
            departure_time: departure.value,
            stops: stops.value,
            destination_city: destination.value,
            "class": travelClass.value,
            duration: parseFloat(duration.value),
            days_left: parseInt(daysLeftInput.value, 10)
        };

        const price = predictPrice(flight);
        resultPrice.textContent = "₹ " + price.toLocaleString("en-IN", {
            minimumFractionDigits: 2, maximumFractionDigits: 2
        });
        resultBox.hidden = false;
        resultBox.scrollIntoView({ behavior: "smooth", block: "center" });
    });
});
