# Importing Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Loading Dataset
def load_dataset():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
               'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, header=None, names=columns)
    print("Dataset Loaded Successfully.")
    return data

# Step 2: Data Exploration
def explore_data(data):
    print("\n--- Data Examination---")
    print("Dataset Head:")
    print(data.head())
    print("\nDataset Info:")
    print(data.info())
    print("\nDataset Description:")
    print(data.describe())
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlations")
    plt.show()

# Step 3: Preprocessing
def preprocess_data(data):
    columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[columns_to_replace] = data[columns_to_replace].replace(0, np.nan)
    data.fillna(data.mean(), inplace=True)
    scaler = StandardScaler()
    features = data.iloc[:, :-1]
    labels = data['Outcome']
    features_scaled = scaler.fit_transform(features)
    print("Data Preprocessing Completed.")
    return features_scaled, labels, scaler

# Step 4: Splitting
def split_data(features, labels, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    print("Data Splitting Completed.")
    return X_train, X_test, y_train, y_test

# Step 5: Training Classifier
def train_classifier(X_train, y_train):
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    print("Model Training Completed.")
    return classifier

# Step 6: Testing & Processing Results
def evaluate_model(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    return accuracy

# Step 8: Application Phase
def predict_diabetes(input_data, classifier, scaler):
    input_array = np.array(input_data).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = classifier.predict(scaled_input)
    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"

# Additional: Feature Importance
def feature_importance(classifier, feature_names):
    importance = classifier.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette="viridis")
    plt.title("Feature Importance")
    plt.show()

# Main Function
def main():
    # Step 1: Load Data
    data = load_dataset()
    
    # Step 2: Explore Data
    explore_data(data)
    
    # Step 3: Preprocess Data
    features_scaled, labels, scaler = preprocess_data(data)
    feature_names = data.columns[:-1]
    
    # Step 4: Split Data
    X_train, X_test, y_train, y_test = split_data(features_scaled, labels)
    
    # Step 5: Train Model
    classifier = train_classifier(X_train, y_train)
    
    # Step 6: Evaluate Model
    accuracy = evaluate_model(classifier, X_test, y_test)
    
    # Step 7: Feature Importance
    feature_importance(classifier, feature_names)
    
    # Step 8: Application Phase
    print("\n--- Application Phase ---")
    example_patient = [2, 120, 75, 30, 90, 33.6, 0.627, 50]
    prediction = predict_diabetes(example_patient, classifier, scaler)
    print(f"Prediction for Example Patient: {prediction}")
    
    # Additional Functionality: Interactive Prediction
    while True:
        print("\nEnter Patient Data or 'exit' to stop:")
        try:
            user_input = input("Format: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age\n> ")
            if user_input.lower() == 'exit':
                print("Exiting the application. Goodbye!")
                break
            patient_data = list(map(float, user_input.split(',')))
            if len(patient_data) != 8:
                raise ValueError("Please enter exactly 8 values!")
            print(f"Prediction: {predict_diabetes(patient_data, classifier, scaler)}")
        except Exception as e:
            print(f"Error: {e}")

# Execute Main Function
if __name__ == "__main__":
    main()
