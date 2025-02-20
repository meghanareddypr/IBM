from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

# Define the Flask app before using @app.route()
app = Flask(__name__)

# Load the trained model
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return render_template('index.html', prediction_text="Model not loaded correctly.")

        # Ensure feature names match the trained model
        feature_names = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]  # Replace with actual feature names
        features = [float(x) for x in request.form.values()]
        features_df = pd.DataFrame([features], columns=feature_names)

        # Make prediction
        prediction = model.predict(features_df)

        # Ensure prediction contains both heating and cooling loads
        if isinstance(prediction, np.ndarray) and prediction.ndim == 2 and prediction.shape[1] == 2:
            heating_load, cooling_load = prediction[0]
        else:
            return render_template('index.html', prediction_text="Unexpected prediction format.")

        # Format the result to show both heating and cooling loads
        heating_load_str = f"{heating_load:.2f}" if isinstance(heating_load, (int, float)) else "N/A"
        cooling_load_str = f"{cooling_load:.2f}" if isinstance(cooling_load, (int, float)) else "N/A"
        result = f'Heating Load: {heating_load_str}, Cooling Load: {cooling_load_str}'

        # Display the result on the webpage
        return render_template('index.html', prediction_text=result)

    except ValueError as ve:
        return render_template('index.html', prediction_text=f"Value Error: {str(ve)}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)




