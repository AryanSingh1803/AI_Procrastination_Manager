from flask import Flask, request, jsonify, render_template
import joblib
import os
import numpy as np

app = Flask(__name__, static_folder="static")

# Load the model
MODEL_PATH = "procrastination_model.pkl"
if os.path.exists(MODEL_PATH):
    print("✅ Model file found!")
    model = joblib.load(MODEL_PATH)
else:
    print("❌ Model file NOT found!")
    model = None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"})

    try:
        # Get input values
        features = []
        for i in range(5):
            value = request.form.get(f"feature{i+1}")

            # Check for missing values
            if value is None or value.strip() == "":
                return jsonify({"error": f"Feature {i+1} is missing!"})

            features.append(float(value))  # Convert to float

        # Reshape input for prediction
        input_array = np.array(features).reshape(1, -1)

        # Predict
        prediction = model.predict(input_array)[0]
        result = "High Procrastination" if prediction == 1 else "Low Procrastination"

        return jsonify({"prediction": result})
    except ValueError:
        return jsonify({"error": "Invalid input! Please enter numerical values."})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
