
let explorerData;

let airlineChart;
let destinationChart;
let daysChart;

// Load the JSON data
async function loadExplorerData() {

    try {

        const response = await fetch("data/explorer.json");

        if (!response.ok) {
            throw new Error("Could not load explorer.json");
        }

        explorerData = await response.json();

        populateSourceCities();

        updateCharts();

    } catch (error) {

        console.error(error);

        document.querySelector(".container").innerHTML =
            "<p>Unable to load fare data. Please refresh the page.</p>";
    }
}


// Populate source city dropdown
function populateSourceCities() {

    const sourceSelect = document.getElementById("sourceFilter");

    sourceSelect.innerHTML = "";

    explorerData.source_cities.forEach(city => {

        const option = document.createElement("option");

        option.value = city;
        option.textContent = city;

        sourceSelect.appendChild(option);

    });
}


// Update all charts
function updateCharts() {

    const selectedClass =
        document.getElementById("classFilter").value;

    const selectedSource =
        document.getElementById("sourceFilter").value;


    // Filter airline data
    const airlineData = explorerData.airline_prices.filter(row =>
        row.class === selectedClass &&
        row.source_city === selectedSource
    );


    // Filter destination data
    const destinationData = explorerData.destination_prices.filter(row =>
        row.class === selectedClass &&
        row.source_city === selectedSource
    );


    // Filter days-left data
    const daysData = explorerData.days_left_prices.filter(row =>
        row.class === selectedClass
    );


    // Sort airline prices
    airlineData.sort((a, b) => a.avg_price - b.avg_price);


    // Sort destination prices
    destinationData.sort((a, b) => a.avg_price - b.avg_price);


    // Sort days
    daysData.sort((a, b) => a.days_left - b.days_left);


    drawAirlineChart(airlineData);

    drawDestinationChart(destinationData);

    drawDaysChart(daysData);

}


// Airline chart
function drawAirlineChart(data) {

    if (airlineChart) {
        airlineChart.destroy();
    }

    airlineChart = new Chart(
        document.getElementById("airlineChart"),
        {
            type: "bar",

            data: {
                labels: data.map(row => row.airline),

                datasets: [{
                    label: "Average Price (₹)",

                    data: data.map(row => row.avg_price),

                    backgroundColor: "#3b82f6",

                    borderRadius: 6
                }]
            },

            options: {
                responsive: true,
                maintainAspectRatio: false,

                plugins: {
                    legend: {
                        display: false
                    }
                },

                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: "Average Price (₹)"
                        }
                    }
                }
            }
        }
    );

}


// Destination chart
function drawDestinationChart(data) {

    if (destinationChart) {
        destinationChart.destroy();
    }

    destinationChart = new Chart(
        document.getElementById("destinationChart"),
        {
            type: "bar",

            data: {
                labels: data.map(row => row.destination_city),

                datasets: [{
                    label: "Average Price (₹)",

                    data: data.map(row => row.avg_price),

                    backgroundColor: "#10b981",

                    borderRadius: 6
                }]
            },

            options: {
                responsive: true,
                maintainAspectRatio: false,

                plugins: {
                    legend: {
                        display: false
                    }
                },

                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        }
    );

}


// Days-left chart
function drawDaysChart(data) {

    if (daysChart) {
        daysChart.destroy();
    }

    daysChart = new Chart(
        document.getElementById("daysChart"),
        {
            type: "line",

            data: {
                labels: data.map(row => row.days_left),

                datasets: [{
                    label: "Average Price (₹)",

                    data: data.map(row => row.avg_price),

                    borderColor: "#8b5cf6",

                    backgroundColor: "rgba(139,92,246,0.15)",

                    fill: true,

                    tension: 0.3
                }]
            },

            options: {
                responsive: true,
                maintainAspectRatio: false,

                scales: {
                    x: {
                        title: {
                            display: true,
                            text: "Days Left Before Departure"
                        }
                    },

                    y: {
                        beginAtZero: true
                    }
                }
            }
        }
    );

}


// Listen for filter changes
document.getElementById("classFilter")
    .addEventListener("change", updateCharts);

document.getElementById("sourceFilter")
    .addEventListener("change", updateCharts);


// Start
loadExplorerData();
