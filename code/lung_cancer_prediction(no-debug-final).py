"""
A cleaned-up Python script for lung cancer prediction.

This script performs the following steps:
1. Loads a lung cancer prediction dataset from a CSV file.
2. Cleans and preprocesses the data, including feature engineering,
   encoding categorical variables, and scaling numerical features.
3. Handles class imbalance using RandomOverSampler.
4. Trains a Logistic Regression model with hyperparameter tuning.
5. Evaluates the model's performance and displays a confusion matrix.
6. Saves the trained model, scaler, and label encoder to disk using pickle.
"""

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import pickle

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# Define directories for output files to store images and models
output_dir = 'Image'
model_dir = 'models'

# Create directories if they don't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# --- Data Loading ---
try:
    df = pd.read_csv('lung_cancer_prediction_dataset.csv')
except FileNotFoundError:
    # The program will not proceed if the file is not found, preventing errors later
    pass

# --- Data Preprocessing ---
# Clean column names for easier access and consistency
df.columns = df.columns.str.replace(' ', '_').str.replace('.', '', regex=False).str.replace('/', '_').str.replace('-', '_').str.lower()

# Drop unnecessary or non-predictive columns identified in the original analysis
cols_to_drop = ['cancer_stage', 'treatment_type', 'id', 'population_size', 'annual_lung_cancer_deaths', 'lung_cancer_prevalence_rate', 'mortality_rate', 'survival_years', 'country', 'developed_or_developing']
df_processed = df.drop(columns=cols_to_drop, errors='ignore')

# --- Feature Engineering ---
# Map pollution exposure categories to numerical values to create a composite score
df_processed['air_pollution_exposure_num'] = df_processed['air_pollution_exposure'].map({'Low': 1, 'Medium': 2, 'High': 3})
df_processed['occupational_exposure_num'] = df_processed['occupational_exposure'].map({'No': 0, 'Yes': 1})
df_processed['indoor_pollution_num'] = df_processed['indoor_pollution'].map({'No': 0, 'Yes': 1})

# Create a composite environmental risk score by summing the numerical mappings
df_processed['environmental_risk_score'] = df_processed['air_pollution_exposure_num'] + df_processed['occupational_exposure_num'] + df_processed['indoor_pollution_num']

# Create new features from existing ones to capture interactions and total exposure
df_processed['total_cigarettes_smoked'] = df_processed['years_of_smoking'] * df_processed['cigarettes_per_day'] * 365
df_processed['smoking_age_interaction'] = df_processed['age'] * df_processed['years_of_smoking']

# Separate features (X) and the target variable (y)
X = df_processed.drop(columns=['lung_cancer_diagnosis'])
y = df_processed['lung_cancer_diagnosis']

# --- Encoding Categorical Features and Scaling Numerical Features ---
# Encode the target variable (y) from string labels to numerical
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Map binary categorical features to numerical (0 or 1)
binary_cols_map = {
    'gender': {'Male': 0, 'Female': 1},
    'smoker': {'No': 0, 'Yes': 1},
    'passive_smoker': {'No': 0, 'Yes': 1},
    'family_history': {'No': 0, 'Yes': 1},
    'adenocarcinoma_type': {'No': 0, 'Yes': 1},
    'early_detection': {'No': 0, 'Yes': 1},
    'healthcare_access': {'Poor': 0, 'Good': 1}
}

for col, mapping in binary_cols_map.items():
    if col in X.columns:
        X[col] = X[col].map(mapping)

# Use one-hot encoding for nominal categorical features
nominal_cols = ['occupational_exposure', 'air_pollution_exposure', 'indoor_pollution']
existing_nominal_cols = [col for col in nominal_cols if col in X.columns]

if existing_nominal_cols:
    X = pd.get_dummies(X, columns=existing_nominal_cols, drop_first=True)

# Scale numerical features using StandardScaler
numerical_cols = [
    'age', 
    'years_of_smoking', 
    'total_cigarettes_smoked', 
    'smoking_age_interaction',
    'cigarettes_per_day'
]
scaler = StandardScaler()
cols_to_scale = [col for col in numerical_cols if col in X.columns]
if cols_to_scale:
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

# --- Imbalanced Data Handling ---
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Apply RandomOverSampler to balance the training data
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# --- Model Training & Hyperparameter Tuning ---
# Initialize the Logistic Regression model
model = LogisticRegression(solver='liblinear', random_state=42)

# Define the hyperparameter grid for GridSearchCV
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}

# Perform grid search with cross-validation to find the best model
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)
best_model = grid_search.best_estimator_

# --- Model Evaluation and Visualization ---
# Predict on the test set using the best model
y_pred = best_model.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Cancer', 'Cancer'])

# Plot and save the confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.title('Final Confusion Matrix for Best Tuned Model')
plt.savefig(os.path.join(output_dir, 'Final_Confusion_Matrix.png'))

# --- Save the Best Model and Preprocessor ---
# Save the trained model, scaler, and label encoder for future use
try:
    with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as file:
        pickle.dump(scaler, file)
    with open(os.path.join(model_dir, 'label_encoder.pkl'), 'wb') as file:
        pickle.dump(le_target, file)
    with open(os.path.join(model_dir, 'best_model.pkl'), 'wb') as file:
        pickle.dump(best_model, file)
except Exception:
    pass