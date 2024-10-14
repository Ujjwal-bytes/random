import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
PlayTennis = pd.read_csv(r"C:\Users\ujjwal jha\Downloads\ai-perform.csv")
print(PlayTennis, "\n\n")

# Apply Label Encoding to transform categorical data into numeric data
PlayTennis = PlayTennis.apply(LabelEncoder().fit_transform)
print(PlayTennis)

# Define feature variables (X) and target variable (y)
X = PlayTennis.iloc[:, :-1]
y = PlayTennis['Play Tennis']

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# Predict on the same data and compare predictions with actual values
X_pred = clf.predict(X)
print(X_pred == y)