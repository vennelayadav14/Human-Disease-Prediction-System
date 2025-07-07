# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler

# 1. Read the dataset
data = pd.read_csv('improved_disease_dataset.csv')

# 2. Encode the 'disease' labels
encoder = LabelEncoder()
data["disease"] = encoder.fit_transform(data["disease"])

# 3. Split features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 4. Balance the dataset
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# 5. Preprocessing before model training
if 'gender' in X_resampled.columns:
    le = LabelEncoder()
    X_resampled['gender'] = le.fit_transform(X_resampled['gender'])

X_resampled = X_resampled.fillna(0)

if len(y_resampled.shape) > 1:
    y_resampled = y_resampled.values.ravel()

# 6. Define models (added SVM here)
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(kernel='linear', probability=True)
}

# 7. Set up Stratified K-Fold
cv_scoring = 'accuracy'
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 8. Cross-validate each model
for model_name, model in models.items():
    try:
        scores = cross_val_score(
            model,
            X_resampled,
            y_resampled,
            cv=stratified_kfold,
            scoring=cv_scoring,
            n_jobs=-1,
            error_score='raise'
        )
        print("=" * 50)
        print(f"Model: {model_name}")
        print(f"Scores: {scores}")
        print(f"Mean Accuracy: {scores.mean():.4f}")
    except Exception as e:
        print("=" * 50)
        print(f"Model: {model_name} failed with error:")
        print(e)
