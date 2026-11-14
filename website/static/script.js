
document.addEventListener("DOMContentLoaded", function () {

    // ======================================
    // 1. GET FORM ELEMENTS
    // ======================================

    const source = document.querySelector("#source_city");
    const destination = document.querySelector("#destination_city");
    const stops = document.querySelector("#stops");
    const airline = document.querySelector("#airline");
    const departure = document.querySelector("#departure_time");
    const duration = document.querySelector("#duration");

    const flightForm = document.querySelector(".flight-form");
    const predictButton = document.querySelector(".predict-button");

    const daysLeftInput = document.querySelector(
        'input[name="days_left"]'
    );

    // Previously submitted selections from Flask
    const saved = {
    airline: document.querySelector("#airline").dataset.selected || "",
    departure: document.querySelector("#departure_time").dataset.selected || "",
    duration: document.querySelector("#duration").dataset.selected || ""
};


    // ======================================
    // 2. RESET DROPDOWNS
    // ======================================

    function resetDropdown(dropdown, message) {

        dropdown.innerHTML = "";

        const option = document.createElement("option");
        option.value = "";
        option.textContent = message;

        dropdown.appendChild(option);
        dropdown.disabled = true;

    }


    // ======================================
    // 3. CHECK SOURCE AND DESTINATION
    // ======================================

    function updateStops() {

        stops.value = "";

        resetDropdown(airline, "Select stops first");
        resetDropdown(departure, "Select airline first");
        resetDropdown(duration, "Select departure time first");

        if (!source.value || !destination.value) {
            stops.disabled = true;
            return;
        }

        if (source.value === destination.value) {

            source.setCustomValidity(
                "Source and destination cannot be the same."
            );

            destination.setCustomValidity(
                "Please select a different destination."
            );

            stops.disabled = true;
            return;
        }

        source.setCustomValidity("");
        destination.setCustomValidity("");

        stops.disabled = false;

    }


    // ======================================
    // 4. LOAD AVAILABLE AIRLINES
    // ======================================

    async function updateAirlines(
        selectedAirline = "",
        selectedDeparture = "",
        selectedDuration = ""
    ) {

        resetDropdown(airline, "Loading airlines...");
        resetDropdown(departure, "Select airline first");
        resetDropdown(duration, "Select departure time first");

        if (!source.value || !destination.value || !stops.value) {
            return;
        }

        try {

            const url =
                `/get-airlines?source=${encodeURIComponent(source.value)}` +
                `&destination=${encodeURIComponent(destination.value)}` +
                `&stops=${encodeURIComponent(stops.value)}`;

            const response = await fetch(url);

            if (!response.ok) {
                throw new Error("Unable to retrieve airlines.");
            }

            const airlines = await response.json();

            resetDropdown(airline, "Select an airline");

            if (airlines.length === 0) {
                resetDropdown(airline, "No airlines available");
                return;
            }

            airlines.forEach(function (name) {

                const option = document.createElement("option");

                option.value = name;
                option.textContent = name.replaceAll("_", " ");

                airline.appendChild(option);

            });

            airline.disabled = false;

            // Restore previously selected airline
            if (selectedAirline && airlines.includes(selectedAirline)) {

                airline.value = selectedAirline;

                await updateDepartureTimes(
                    selectedDeparture,
                    selectedDuration
                );

            }

        } catch (error) {

            console.error(error);
            resetDropdown(airline, "Unable to load airlines");

        }

    }


    // ======================================
    // 5. LOAD AVAILABLE DEPARTURE TIMES
    // ======================================

    async function updateDepartureTimes(
        selectedDeparture = "",
        selectedDuration = ""
    ) {

        resetDropdown(departure, "Loading departure times...");
        resetDropdown(duration, "Select departure time first");

        if (!airline.value) {
            return;
        }

        try {

            const url =
                `/get-departure-times?source=${encodeURIComponent(source.value)}` +
                `&destination=${encodeURIComponent(destination.value)}` +
                `&stops=${encodeURIComponent(stops.value)}` +
                `&airline=${encodeURIComponent(airline.value)}`;

            const response = await fetch(url);

            if (!response.ok) {
                throw new Error("Unable to retrieve departure times.");
            }

            const times = await response.json();

            resetDropdown(departure, "Select departure time");

            const labels = {
                "Early_Morning": "Early Morning (5 AM – 8 AM)",
                "Morning": "Morning (8 AM – 12 PM)",
                "Afternoon": "Afternoon (12 PM – 4 PM)",
                "Evening": "Evening (4 PM – 8 PM)",
                "Night": "Night (8 PM – 12 AM)",
                "Late_Night": "Late Night (12 AM – 5 AM)"
            };

            if (times.length === 0) {
                resetDropdown(departure, "No departure times available");
                return;
            }

            times.forEach(function (time) {

                const option = document.createElement("option");

                option.value = time;
                option.textContent = labels[time] || time;

                departure.appendChild(option);

            });

            departure.disabled = false;

            // Restore previously selected departure time
            if (selectedDeparture && times.includes(selectedDeparture)) {

                departure.value = selectedDeparture;

                await updateDurations(selectedDuration);

            }

        } catch (error) {

            console.error(error);
            resetDropdown(departure, "Unable to load departure times");

        }

    }


    // ======================================
    // 6. LOAD AVAILABLE FLIGHT DURATIONS
    // ======================================

    async function updateDurations(selectedDuration = "") {

        resetDropdown(duration, "Loading durations...");

        if (!departure.value) {
            return;
        }

        try {

            const url =
                `/get-durations?source=${encodeURIComponent(source.value)}` +
                `&destination=${encodeURIComponent(destination.value)}` +
                `&stops=${encodeURIComponent(stops.value)}` +
                `&airline=${encodeURIComponent(airline.value)}` +
                `&departure_time=${encodeURIComponent(departure.value)}`;

            const response = await fetch(url);

            if (!response.ok) {
                throw new Error("Unable to retrieve durations.");
            }

            const durations = await response.json();

            resetDropdown(duration, "Select flight duration");

            if (durations.length === 0) {
                resetDropdown(duration, "No durations available");
                return;
            }

            durations.forEach(function (hours) {

                const option = document.createElement("option");

                option.value = hours;
                option.textContent = `${hours} hours`;

                duration.appendChild(option);

            });

            duration.disabled = false;

            // Restore previously selected duration
            if (selectedDuration !== "") {

                const matchingOption = Array.from(duration.options).find(
                    option => Number(option.value) === Number(selectedDuration)
                );

                if (matchingOption) {
                    duration.value = matchingOption.value;
                }

            }

        } catch (error) {

            console.error(error);
            resetDropdown(duration, "Unable to load durations");

        }

    }


    // ======================================
    // 7. RESTORE PREVIOUS SELECTIONS
    // ======================================

    async function restoreSelections() {

        if (!source.value || !destination.value) {
            return;
        }

        if (source.value === destination.value) {

            source.setCustomValidity(
                "Source and destination cannot be the same."
            );

            destination.setCustomValidity(
                "Please select a different destination."
            );

            stops.disabled = true;
            return;
        }

        source.setCustomValidity("");
        destination.setCustomValidity("");

        stops.disabled = false;

        if (stops.value) {

            await updateAirlines(
                saved.airline,
                saved.departure,
                saved.duration
            );

        }

    }


    // ======================================
    // 8. HANDLE DROPDOWN CHANGES
    // ======================================

    source.addEventListener("change", updateStops);

    destination.addEventListener("change", updateStops);

    stops.addEventListener("change", function () {
        updateAirlines();
    });

    airline.addEventListener("change", function () {
        updateDepartureTimes();
    });

    departure.addEventListener("change", function () {
        updateDurations();
    });


    // ======================================
    // 9. VALIDATE DAYS BEFORE DEPARTURE
    // ======================================

    daysLeftInput.addEventListener("input", function () {

        const days = Number(daysLeftInput.value);

        if (days < 1 || days > 49) {

            daysLeftInput.setCustomValidity(
                "Days before departure must be between 1 and 49."
            );

        } else {

            daysLeftInput.setCustomValidity("");

        }

    });


    // ======================================
    // 10. FORM SUBMISSION
    // ======================================

    flightForm.addEventListener("submit", function (event) {

        if (source.value === destination.value) {

            event.preventDefault();

            destination.setCustomValidity(
                "Source and destination cannot be the same."
            );

            destination.reportValidity();

            return;
        }

        source.setCustomValidity("");
        destination.setCustomValidity("");

        if (!flightForm.checkValidity()) {

            event.preventDefault();
            return;

        }

        predictButton.disabled = true;
        predictButton.textContent = "Predicting Your Fare...";

    });


    // Restore selections when the prediction page loads
    restoreSelections();

});
