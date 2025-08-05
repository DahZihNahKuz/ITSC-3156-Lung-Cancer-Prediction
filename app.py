import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing for the frontend

# --- This section is for creating a dummy model. Uncomment and run this file once to generate the model.joblib file. ---
"""
# Create a dummy dataset to train a simple model for demonstration purposes
data = {
    'Age': [45, 65, 30, 50, 70, 25, 60, 40],
    'Gender_Male': [1, 1, 0, 0, 1, 0, 1, 0],
    'Smoker_Yes': [1, 1, 0, 1, 1, 0, 1, 0],
    'Years_of_Smoking': [20, 40, 0, 15, 50, 0, 30, 0],
    'Cigarettes_per_Day': [15, 30, 0, 10, 25, 0, 20, 0],
    'Cancer': [0, 1, 0, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Define features (X) and target (y)
X = df[['Age', 'Gender_Male', 'Smoker_Yes', 'Years_of_Smoking', 'Cigarettes_per_Day']]
y = df['Cancer']

# Train a simple Logistic Regression model
model = LogisticRegression(random_state=0)
model.fit(X, y)

# Save the trained model to a file
joblib.dump(model, 'model.joblib')
print("Dummy model created and saved as model.joblib")
"""
# --- End of dummy model creation section. Make sure to comment this out after running once. ---


# Load the pre-trained model
MODEL_PATH = 'model.joblib'
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found. Please train and save the model first.")
    exit()

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)

        # Create a pandas DataFrame from the input data
        # Ensure the column order and names match the model's training data
        # Handle one-hot encoding for 'Gender' and 'Smoker'
        gender_male = 1 if data.get('gender') == 'Male' else 0
        smoker_yes = 1 if data.get('smoker') == 'Yes' else 0

        features = pd.DataFrame([[
            data.get('age'),
            gender_male,
            smoker_yes,
            data.get('yearsSmoking'),
            data.get('cigsPerDay')
        ]], columns=['Age', 'Gender_Male', 'Smoker_Yes', 'Years_of_Smoking', 'Cigarettes_per_Day'])
        
        # Make the prediction
        prediction = model.predict(features)
        
        # Return the prediction as a JSON response
        result = int(prediction[0]) # Convert numpy.int64 to standard Python int
        
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask application
if __name__ == '__main__':
    # You can change the port if needed. Host is set to '0.0.0.0' to be accessible from other devices on the network.
    app.run(host='0.0.0.0', port=5000, debug=True)
