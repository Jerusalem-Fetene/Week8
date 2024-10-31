from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
with open('fraud_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Data Preprocessing Function
def preprocess_fraud_data(data):
    # Ensure all required features are present in the input data
    required_features = model.feature_names_in_
    for feature in required_features:
        if feature not in data:
            data[feature] = 0

    # Convert data into DataFrame
    data_df = pd.DataFrame([data])
    
    # Reorder columns to match the training data
    data_df = data_df[required_features]

    return data_df

@app.route('/')
def home():
    return "Fraud Detection Model API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        app.logger.info(f"Received data: {data}")  # Log incoming data for debugging

        # Preprocess incoming data
        data_df = preprocess_fraud_data(data)

        # Make prediction using the loaded model
        prediction = model.predict(data_df)
        app.logger.info(f"Prediction result: {prediction}")

        # Prepare response as JSON
        response = {'prediction': int(prediction[0])}  # Assuming prediction is binary (0 or 1)

        return jsonify(response)

    except KeyError as e:
        app.logger.error(f"KeyError: {str(e)}")
        return jsonify({'error': f'KeyError: {str(e)}'}), 400  # HTTP status 400 for bad request
    except ValueError as e:
        app.logger.error(f"ValueError: {str(e)}")
        return jsonify({'error': f'ValueError: {str(e)}'}), 400  # HTTP status 400 for bad request
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'Unexpected error occurred'}), 500  # Return error message and HTTP status 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
