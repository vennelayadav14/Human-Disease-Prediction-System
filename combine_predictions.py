# combine_predictions.py

import pickle
from statistics import mode
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved predictions
with open('svm_preds.pkl', 'rb') as f:
    svm_preds = pickle.load(f)

with open('nb_preds.pkl', 'rb') as f:
    nb_preds = pickle.load(f)

with open('rf_preds.pkl', 'rb') as f:
    rf_preds = pickle.load(f)

# Load true labels
with open('y_resampled.pkl', 'rb') as f:
    y_resampled = pickle.load(f)

# Combine predictions using majority voting
final_preds = [mode([i, j, k]) for i, j, k in zip(svm_preds, nb_preds, rf_preds)]

# Generate confusion matrix
cf_matrix_combined = confusion_matrix(y_resampled, final_preds)

# Plot the confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix_combined, annot=True, fmt="d", cmap='Blues')
plt.title("Confusion Matrix for Combined Model")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Print Combined Model Accuracy
accuracy = accuracy_score(y_resampled, final_preds) * 100
print(f"Combined Model Accuracy: {accuracy:.2f}%")