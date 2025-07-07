# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler

# Reading the dataset
data = pd.read_csv('improved_disease_dataset.csv')

# Encode target column
encoder = LabelEncoder()
data["disease"] = encoder.fit_transform(data["disease"])

# Split features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Visualize the class distribution
plt.figure(figsize=(18, 8))
sns.countplot(x=y)
plt.title("Disease Class Distribution Before Resampling")
plt.xticks(rotation=90)
plt.show()

# Balance the dataset
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Show resampled class distribution
print("Resampled Class Distribution:\n", pd.Series(y_resampled).value_counts())
