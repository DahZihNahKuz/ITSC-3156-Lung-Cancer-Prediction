import os
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import RandomOverSampler
import pickle
import matplotlib.pyplot as plt

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# Define directories for output files to store models
model_dir = 'models'

# Create directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# --- Data Loading ---
try:
    # This assumes you have the dataset available in the same directory.
    df = pd.read_csv('lung_cancer_prediction_dataset.csv')
except FileNotFoundError:
    print("Error: Dataset file 'lung_cancer_prediction_dataset.csv' not found.")
    exit()

# --- Data Preprocessing & Feature Engineering ---
# Clean column names for consistency
df.columns = df.columns.str.replace(' ', '_').str.replace('.', '', regex=False).str.replace('/', '_').str.replace('-', '_').str.lower()

# Drop unnecessary columns
cols_to_drop = [
    'cancer_stage', 'treatment_type', 'id', 'population_size',
    'annual_lung_cancer_deaths', 'lung_cancer_prevalence_rate',
    'mortality_rate', 'survival_years', 'country',
    'developed_or_developing', 'adenocarcinoma_type', 'early_detection'
]
df_processed = df.drop(columns=cols_to_drop, errors='ignore')

# Create feature engineering columns first
# These numerical columns are created for the composite score and will be dropped later.
df_processed['air_pollution_exposure_num'] = df_processed['air_pollution_exposure'].map({'Low': 1, 'Medium': 2, 'High': 3})
df_processed['occupational_exposure_num'] = df_processed['occupational_exposure'].map({'No': 0, 'Yes': 1})
df_processed['indoor_pollution_num'] = df_processed['indoor_pollution'].map({'No': 0, 'Yes': 1})
df_processed['environmental_risk_score'] = df_processed['air_pollution_exposure_num'] + df_processed['occupational_exposure_num'] + df_processed['indoor_pollution_num']
df_processed['total_cigarettes_smoked'] = df_processed['years_of_smoking'] * df_processed['cigarettes_per_day'] * 365
df_processed['smoking_age_interaction'] = df_processed['age'] * df_processed['years_of_smoking']

# Separate features (X) and the target variable (y)
X = df_processed.drop(columns=['lung_cancer_diagnosis'])
y = df_processed['lung_cancer_diagnosis']

# --- Encoding Categorical Features and Scaling Numerical Features ---
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Map binary categorical features to numerical (0 or 1)
binary_cols_map = {
    'gender': {'Male': 0, 'Female': 1},
    'smoker': {'No': 0, 'Yes': 1},
    'passive_smoker': {'No': 0, 'Yes': 1},
    'family_history': {'No': 0, 'Yes': 1},
    'healthcare_access': {'Poor': 0, 'Good': 1}
}
for col, mapping in binary_cols_map.items():
    if col in X.columns:
        X[col] = X[col].map(mapping)

# Use one-hot encoding for nominal categorical features.
# This is a key step to ensure consistency with the new backend.
nominal_cols = ['occupational_exposure', 'air_pollution_exposure', 'indoor_pollution']
existing_nominal_cols = [col for col in nominal_cols if col in X.columns]
if existing_nominal_cols:
    X = pd.get_dummies(X, columns=existing_nominal_cols, drop_first=True)

# Drop redundant numerical columns created for composite scores
# This is where the old model and new model diverge. The old model was trained on these columns.
X = X.drop(columns=['air_pollution_exposure_num', 'occupational_exposure_num', 'indoor_pollution_num'], errors='ignore')

# Scale numerical features
numerical_cols = ['age', 'years_of_smoking', 'total_cigarettes_smoked', 'smoking_age_interaction', 'cigarettes_per_day']
scaler = StandardScaler()
cols_to_scale = [col for col in numerical_cols if col in X.columns]
if cols_to_scale:
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

# --- Imbalanced Data Handling ---
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# --- Model Training & Hyperparameter Tuning ---
model = LogisticRegression(solver='liblinear', random_state=42)
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)
best_model = grid_search.best_estimator_

# --- Model Evaluation & Saving ---
y_pred = best_model.predict(X_test)

try:
    with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as file:
        pickle.dump(scaler, file)
    with open(os.path.join(model_dir, 'label_encoder.pkl'), 'wb') as file:
        pickle.dump(le_target, file)
    with open(os.path.join(model_dir, 'best_model.pkl'), 'wb') as file:
        pickle.dump(best_model, file)
    print("Model files saved successfully.")
except Exception as e:
    print(f"Error saving files: {e}")
