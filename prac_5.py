import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset (replace with your dataset path)
df = pd.read_csv(r"C:\Users\ujjwal jha\Downloads\ai-perform.csv")
print(df, "\n\n")

# Apply Label Encoding to convert categorical data to numerical data
df = df.apply(LabelEncoder().fit_transform)

# Define features (X) and target variable (y)
X = df.iloc[:, :-1]  # All columns except the target
y = df['Play Tennis']  # Target column

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier
svm_clf = SVC(kernel='linear')  # Using a linear kernel

# Train the model
svm_clf.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Compare predictions with actual values
print(f"Predictions: {y_pred}")
print(f"Actual: {y_test.values}")