
let flyWiseModel = null;

// Load the exported model
async function loadFlyWiseModel() {
    if (flyWiseModel) {
        return flyWiseModel;
    }

    const response = await fetch("model.json");

    if (!response.ok) {
        throw new Error("Could not load the FlyWise model.");
    }

    flyWiseModel = await response.json();
    return flyWiseModel;
}

// Convert flight details into the model's feature format
function prepareFlightFeatures(model, flight) {
    const features = [];

    const categoricalFields = [
        "airline",
        "source_city",
        "departure_time",
        "stops",
        "arrival_time",
        "destination_city",
        "class"
    ];

    // One-hot encode categorical features
    for (const field of categoricalFields) {
        const categories = model.categories[field];

        if (!categories.includes(flight[field])) {
            throw new Error("Invalid value for " + field);
        }

        for (const category of categories) {
            features.push(category === flight[field] ? 1 : 0);
        }
    }

    // Standardize numerical features
    const numericalValues = [
        Number(flight.duration),
        Number(flight.days_left)
    ];

    numericalValues.forEach((value, index) => {
        const standardized =
            (value - model.scaling.means[index]) /
            model.scaling.scales[index];

        features.push(standardized);
    });

    return features;
}

// Convert feature values into histogram bins
function convertToBins(model, features) {
    return features.map((value, index) => {
        const thresholds = model.bin_thresholds[index];

        let low = 0;
        let high = thresholds.length;

        while (low < high) {
            const middle = Math.floor((low + high) / 2);

            if (value < thresholds[middle]) {
                high = middle;
            } else {
                low = middle + 1;
            }
        }

        return low;
    });
}

// Evaluate one decision tree
function predictTree(tree, bins) {
    let nodeIndex = 0;

    while (!tree[nodeIndex].is_leaf) {
        const node = tree[nodeIndex];
        const featureValue = bins[node.feature_idx];

        if (featureValue <= node.bin_threshold) {
            nodeIndex = node.left;
        } else {
            nodeIndex = node.right;
        }
    }

    return tree[nodeIndex].value;
}

// Predict the flight price
async function predictFlyWise(flight) {
    const model = await loadFlyWiseModel();

    const features = prepareFlightFeatures(model, flight);
    const bins = convertToBins(model, features);

    let prediction = model.baseline;

    for (const tree of model.trees) {
        prediction += model.learning_rate * predictTree(tree, bins);
    }

    return Math.max(0, prediction);
}
