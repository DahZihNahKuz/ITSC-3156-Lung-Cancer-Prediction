import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Define directories for model files
model_dir = './models'

# Load the pre-trained model and preprocessors
try:
    with open(os.path.join(model_dir, 'best_model.pkl'), 'rb') as f:
        best_model = pickle.load(f)
    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
        le_target = pickle.load(f)
    print("Model files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Model file not found - {e}. Please ensure the models directory and files exist.")
    best_model = None
    scaler = None
    le_target = None
except Exception as e:
    print(f"An unexpected error occurred while loading models: {e}")
    best_model = None
    scaler = None
    le_target = None

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles prediction requests by preprocessing data and returning a prediction.
    This version is highly resilient to missing or malformed input data by
    explicitly building the feature vector and handling NoneType errors.
    """
    if not all([best_model, scaler, le_target]):
        return jsonify({'status': 'error', 'error': 'Model files not loaded. Please check server logs.'}), 500

    try:
        data = request.get_json()

        # Define the exact list of features expected by the model.
        # This includes all one-hot encoded columns.
        feature_order = [
            'age', 'gender', 'smoker', 'years_of_smoking', 'cigarettes_per_day',
            'passive_smoker', 'family_history', 'healthcare_access',
            'environmental_risk_score', 'total_cigarettes_smoked',
            'smoking_age_interaction', 'occupational_exposure_Yes',
            'air_pollution_exposure_Low', 'air_pollution_exposure_Medium',
            'indoor_pollution_Yes'
        ]

        # Use a dictionary to build the feature vector to prevent NaNs.
        # Initialize all features to 0 or a sensible default.
        processed_data = {
            'age': 0.0, 'gender': 0.0, 'smoker': 0.0, 'years_of_smoking': 0.0,
            'cigarettes_per_day': 0.0, 'passive_smoker': 0.0, 'family_history': 0.0,
            'healthcare_access': 0.0, 'environmental_risk_score': 0.0,
            'total_cigarettes_smoked': 0.0, 'smoking_age_interaction': 0.0,
            'occupational_exposure_Yes': 0.0, 'air_pollution_exposure_Low': 0.0,
            'air_pollution_exposure_Medium': 0.0, 'indoor_pollution_Yes': 0.0
        }

        # --- Data Preprocessing & Feature Engineering (must match train_model.py) ---
        # Robustly process the input data and populate the `processed_data` dictionary.

        # 1. Handle numerical features with a check for NoneType
        age = data.get('age')
        processed_data['age'] = float(age) if age is not None else 0.0

        years_of_smoking = data.get('years_of_smoking')
        processed_data['years_of_smoking'] = float(years_of_smoking) if years_of_smoking is not None else 0.0

        cigarettes_per_day = data.get('cigarettes_per_day')
        processed_data['cigarettes_per_day'] = float(cigarettes_per_day) if cigarettes_per_day is not None else 0.0

        # 2. Handle binary features using a map.
        binary_map = {'Yes': 1, 'No': 0, 'Male': 0, 'Female': 1, 'Poor': 0, 'Good': 1}
        processed_data['gender'] = binary_map.get(data.get('gender', 'Female'), 0)
        processed_data['smoker'] = binary_map.get(data.get('smoker', 'No'), 0)
        processed_data['passive_smoker'] = binary_map.get(data.get('passive_smoker', 'No'), 0)
        processed_data['family_history'] = binary_map.get(data.get('family_history', 'No'), 0)
        processed_data['healthcare_access'] = binary_map.get(data.get('healthcare_access', 'Poor'), 0)
        
        # 3. Handle one-hot encoded features
        if data.get('occupational_exposure') == 'Yes':
            processed_data['occupational_exposure_Yes'] = 1.0
        
        air_pollution_exposure = data.get('air_pollution_exposure')
        if air_pollution_exposure == 'Low':
            processed_data['air_pollution_exposure_Low'] = 1.0
        elif air_pollution_exposure == 'Medium':
            processed_data['air_pollution_exposure_Medium'] = 1.0
            
        if data.get('indoor_pollution') == 'Yes':
            processed_data['indoor_pollution_Yes'] = 1.0

        # 4. Create composite/engineered features
        air_pollution_num = 0
        if air_pollution_exposure == 'Low':
            air_pollution_num = 1
        elif air_pollution_exposure == 'Medium':
            air_pollution_num = 2
        elif air_pollution_exposure == 'High':
            air_pollution_num = 3
        
        occupational_exposure_num = 1 if data.get('occupational_exposure') == 'Yes' else 0
        indoor_pollution_num = 1 if data.get('indoor_pollution') == 'Yes' else 0

        processed_data['environmental_risk_score'] = (
            air_pollution_num + occupational_exposure_num + indoor_pollution_num
        )
        
        # Guard against potential division by zero or errors
        processed_years_of_smoking = processed_data.get('years_of_smoking', 0)
        processed_cigarettes_per_day = processed_data.get('cigarettes_per_day', 0)
        processed_age = processed_data.get('age', 0)
        
        processed_data['total_cigarettes_smoked'] = processed_years_of_smoking * processed_cigarettes_per_day * 365
        processed_data['smoking_age_interaction'] = processed_age * processed_years_of_smoking

        # 5. Create a final DataFrame from the processed dictionary.
        # This guarantees the correct structure and no NaNs.
        final_df = pd.DataFrame([processed_data])[feature_order]

        # Select the numerical columns for scaling
        numerical_cols_to_scale = ['age', 'years_of_smoking', 'total_cigarettes_smoked', 'smoking_age_interaction', 'cigarettes_per_day']

        # Apply the pre-fitted scaler.
        final_df[numerical_cols_to_scale] = scaler.transform(final_df[numerical_cols_to_scale])

        # Make the prediction
        prediction_encoded = best_model.predict(final_df)
        prediction_probabilities = best_model.predict_proba(final_df)[0]

        # Decode the prediction
        prediction = le_target.inverse_transform(prediction_encoded)[0]

        response = {
            'status': 'success',
            'prediction': prediction,
            'probability_no_cancer': round(prediction_probabilities[0], 4),
            'probability_cancer': round(prediction_probabilities[1], 4)
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
