import pickle
import os
import logging
import pandas as pd
from flask import Flask, request, jsonify

# Configuration
model_file = os.environ.get('MODEL_FILE', '../models/LabelEncoder_Pipeline_RandomForest_SerovarClassifier_v1.bin')
port = int(os.environ.get('PORT', 9696))

logging.basicConfig(level=logging.INFO)

# Load model
try:
    with open(model_file, 'rb') as f_in:
        label_encoder, model_pipeline = pickle.load(f_in)
    logging.info(f"Model {model_file} loaded successfully. Ready to recieve requests and make predictions.")
except FileNotFoundError:
    logging.error(f"Model file {model_file} not found.")
    exit(1)

app = Flask('Serovar-Classification')

@app.route('/predict_serovar', methods=['POST'])
def predict_serovar():
    """
    Predicts serovar type of a Salmonella strain based on its nutrient-utilization profile.
    """
    try:
        strain_profile = request.get_json()

        if not strain_profile:
            raise ValueError("No input data provided")

        X_request = pd.DataFrame.from_dict(strain_profile,orient='index').T
        y_pred = model_pipeline.predict(X_request)

        prediction = label_encoder.inverse_transform(y_pred)[0]

        result = {
            'Serovar prediction': prediction,
        }

        return jsonify(result)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=port)