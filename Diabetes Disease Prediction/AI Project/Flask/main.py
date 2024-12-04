from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import logging

# Setup Flask App
app = Flask(__name__)

# Global variables for model and scaler
classifier = None
scaler = None
feature_names = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Dataset
def load_dataset():
    try:
        data = pd.read_csv('pima_diabetes.csv')
        logging.info("Dataset loaded from local file.")
    except FileNotFoundError:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
                   'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        data = pd.read_csv(url, header=None, names=columns)
        data.to_csv('pima_diabetes.csv', index=False)
        logging.info("Dataset downloaded and saved locally.")
    return data

# Add Derived Features
def add_features(data):
    return data

# # Preprocess Data
# def preprocess_data(data):
#     columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
#     data[columns_to_replace] = data[columns_to_replace].replace(0, np.nan)
#     imputer = SimpleImputer(strategy='mean')
#     data[columns_to_replace] = imputer.fit_transform(data[columns_to_replace])
#     data = add_features(data)
#     scaler = StandardScaler()
#     features = data.drop(columns=['Outcome'])
#     labels = data['Outcome']
#     features_scaled = scaler.fit_transform(features)
#     return features_scaled, labels, scaler
# Preprocess Data - Use 8 features only
def preprocess_data(data):
    columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[columns_to_replace] = data[columns_to_replace].replace(0, np.nan)
    imputer = SimpleImputer(strategy='mean')
    data[columns_to_replace] = imputer.fit_transform(data[columns_to_replace])
    # No feature engineering (derived features)
    scaler = StandardScaler()
    features = data.iloc[:, :-1]  # Use only 8 features (no derived features)
    labels = data['Outcome']
    features_scaled = scaler.fit_transform(features)
    return features_scaled, labels, scaler

# Balance the Dataset
def balance_data(features, labels):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(features, labels)
    logging.info("Data Balancing Completed.")
    return X_resampled, y_resampled

# Split Data
def split_data(features, labels, test_size=0.2):
    return train_test_split(features, labels, test_size=test_size, random_state=42)

# Train Ensemble Classifier
def train_ensemble(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    ensemble = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='soft')
    ensemble.fit(X_train, y_train)
    return ensemble

# Feature Importance
def feature_importance(classifier, feature_names):
    if hasattr(classifier, 'feature_importances_'):
        importance = classifier.feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False)
        print("\nFeature Importance:")
        print(importance_df)

# Save Model and Scaler
def save_model(classifier, scaler):
    joblib.dump(classifier, 'classifier.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    logging.info("Model and scaler saved.")

# Load Model and Scaler
def load_model():
    try:
        classifier = joblib.load('classifier.pkl')
        scaler = joblib.load('scaler.pkl')
        logging.info("Model and scaler loaded.")
        return classifier, scaler
    except FileNotFoundError:
        logging.warning("No saved model found. Train the model first.")
        return None, None

# Predict Diabetes
def predict_diabetes(input_data, classifier, scaler):
    input_array = np.array(input_data).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = classifier.predict(scaled_input)
    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"

# Home Route
@app.route('/')
def home():
    return render_template('index.html')
# Train Route
@app.route('/train', methods=['POST'])
def train_model():
    global classifier, scaler, feature_names
    data = load_dataset()
    features_scaled, labels, scaler = preprocess_data(data)
    feature_names = data.columns[:-1]  # 8 original features
    X_resampled, y_resampled = balance_data(features_scaled, labels)
    X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)
    classifier = train_ensemble(X_train, y_train)
    save_model(classifier, scaler)
    accuracy = accuracy_score(y_test, classifier.predict(X_test))
    logging.info(f"Model trained. Accuracy: {accuracy * 100:.2f}%")
    return jsonify({"message": "Model trained successfully!", "accuracy": f"{accuracy * 100:.2f}%"})

# Predict Route
@app.route('/predict', methods=['POST'])
def predict():
    global classifier, scaler
    if not classifier or not scaler:
        return jsonify({"error": "Model not trained yet. Train the model first."}), 400

    input_data = request.json.get('input_data')
    if not input_data or not isinstance(input_data, list) or len(input_data) != 8:
        return jsonify({"error": "Invalid input data. Provide a list of 8 numeric values for prediction."}), 400

    try:
        input_data = list(map(float, input_data))
        prediction = predict_diabetes(input_data, classifier, scaler)
        return jsonify({"prediction": prediction})
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

# Train Route
# @app.route('/train', methods=['POST'])
# def train_model():
#     global classifier, scaler, feature_names
#     data = load_dataset()
#     features_scaled, labels, scaler = preprocess_data(data)
#     feature_names = data.columns[:-1]
#     X_resampled, y_resampled = balance_data(features_scaled, labels)
#     X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)
#     classifier = train_ensemble(X_train, y_train)
#     save_model(classifier, scaler)
#     accuracy = accuracy_score(y_test, classifier.predict(X_test))
#     logging.info(f"Model trained. Accuracy: {accuracy * 100:.2f}%")
#     return jsonify({"message": "Model trained successfully!", "accuracy": f"{accuracy * 100:.2f}%"})

# # Predict Route
# @app.route('/predict', methods=['POST'])
# def predict():
#     global classifier, scaler
#     if not classifier or not scaler:
#         return jsonify({"error": "Model not trained yet. Train the model first."}), 400

#     input_data = request.json.get('input_data')
#     if not input_data or not isinstance(input_data, list) or len(input_data) != 8:
#         return jsonify({"error": "Invalid input data. Provide a list of 8 numeric values for prediction."}), 400

#     try:
#         input_data = list(map(float, input_data))
#         prediction = predict_diabetes(input_data, classifier, scaler)
#         return jsonify({"prediction": prediction})
#     except Exception as e:
#         logging.error(f"Prediction error: {str(e)}")
#         return jsonify({"error": f"Processing error: {str(e)}"}), 500


# Error Handler
@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Internal Server Error: {str(error)}")
    return jsonify({"error": "An internal server error occurred."}), 500

# Load Model on App Startup
classifier, scaler = load_model()

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
