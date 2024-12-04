from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Flask App Setup
app = Flask(__name__, template_folder='templates')

# Global variables for data and model
classifier = None
scaler = None
feature_names = None

# Load Dataset
def load_dataset():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
               'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, header=None, names=columns)
    return data

# Preprocess Data
def preprocess_data(data):
    columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[columns_to_replace] = data[columns_to_replace].replace(0, np.nan)
    data.fillna(data.mean(), inplace=True)
    scaler = StandardScaler()
    features = data.iloc[:, :-1]
    labels = data['Outcome']
    features_scaled = scaler.fit_transform(features)
    return features_scaled, labels, scaler

# Split Data
def split_data(features, labels, test_size=0.2):
    return train_test_split(features, labels, test_size=test_size, random_state=42)

# Train Classifier with Hyperparameter Tuning
def train_classifier(X_train, y_train):
    # Configure the Random Forest with optimized hyperparameters
    classifier = RandomForestClassifier(
        n_estimators=200,        # Number of trees in the forest
        max_depth=10,            # Maximum depth of each tree
        min_samples_split=5,     # Minimum number of samples required to split an internal node
        min_samples_leaf=2,      # Minimum number of samples required at each leaf node
        random_state=42,         # Random state for reproducibility
        max_features='sqrt',     # Number of features to consider at each split
        bootstrap=True           # Whether to use bootstrap samples
    )
    classifier.fit(X_train, y_train)
    return classifier

# Prediction
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
    feature_names = data.columns[:-1]
    X_train, X_test, y_train, y_test = split_data(features_scaled, labels)
    classifier = train_classifier(X_train, y_train)
    accuracy = accuracy_score(y_test, classifier.predict(X_test))
    return jsonify({"message": "Model trained successfully!", "accuracy": f"{accuracy * 100:.2f}%"})

# Predict Route
@app.route('/predict', methods=['POST'])
def predict():
    global classifier, scaler
    if not classifier or not scaler:
        return jsonify({"error": "Model not trained yet. Train the model first."}), 400
    
    input_data = request.json.get('input_data')
    if not input_data or len(input_data) != 8:
        return jsonify({"error": "Invalid input data. Provide 8 values for prediction."}), 400
    
    try:
        prediction = predict_diabetes(input_data, classifier, scaler)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
