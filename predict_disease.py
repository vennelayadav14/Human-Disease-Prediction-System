import numpy as np
import pandas as pd
from statistics import mode
import joblib

# Load the dataset to get the feature names
data = pd.read_csv('improved_disease_dataset.csv')

# Extract features and labels (X: symptoms, y: disease)
X = data.iloc[:, :-1]  # assuming that the last column is 'disease'
y = data.iloc[:, -1]

# Load the trained models
svm_model = joblib.load('svm_model.pkl')
nb_model = joblib.load('nb_model.pkl')
rf_model = joblib.load('rf_model.pkl')

# Load the label encoder
encoder = joblib.load('label_encoder.pkl')

# Get the symptom column names (features)
symptoms = X.columns.values
symptom_index = {symptom: idx for idx, symptom in enumerate(symptoms)}

def predict_disease(input_symptoms):
    # Convert input symptoms to a list
    input_symptoms = input_symptoms.split(",")
    
    # Initialize a zero vector for the input data
    input_data = [0] * len(symptom_index)
    
    # Mark the symptoms in the vector based on input
    for symptom in input_symptoms:
        symptom = symptom.strip()  # Remove any extra spaces
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1
        else:
            print(f"Warning: '{symptom}' is not a valid symptom in the dataset.")
    
    # Convert input data to a numpy array
    input_data = np.array(input_data).reshape(1, -1)

    # Get predictions from each model
    rf_pred = rf_model.predict(input_data)[0]
    nb_pred = nb_model.predict(input_data)[0]
    svm_pred = svm_model.predict(input_data)[0]

    # Convert numeric predictions back to disease labels
    rf_pred_label = encoder.inverse_transform([rf_pred])[0]
    nb_pred_label = encoder.inverse_transform([nb_pred])[0]
    svm_pred_label = encoder.inverse_transform([svm_pred])[0]

    # Combine predictions using majority voting
    final_pred = mode([rf_pred_label, nb_pred_label, svm_pred_label])
    
    return {
        "Random Forest Prediction": rf_pred_label,
        "Naive Bayes Prediction": nb_pred_label,
        "SVM Prediction": svm_pred_label,
        "Final Prediction": final_pred
    }

# Test the prediction function with a sample input
user_input = input("Please enter symptoms separated by commas (e.g., 'Itching, Skin Rash, Nodal Skin Eruptions'): ")
print(predict_disease(user_input))
