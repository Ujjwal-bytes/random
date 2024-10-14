import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Load the data
df = pd.read_csv(r"C:\Users\ujjwal jha\Downloads\diabetes.csv")
print("Dataset:\n", df.head())  # Display the first few rows of the dataset

# Prepare the data
X, y = df.drop('Outcome', axis=1), df['Outcome']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Plot k-NN accuracy
plt.plot(range(1, 9), [KNeighborsClassifier(n).fit(X_train, y_train).score(X_test, y_test) for n in range(1, 9)])
plt.title('k-NN Accuracy')
plt.xlabel('Neighbors')
plt.ylabel('Accuracy')
plt.show()

# Grid Search for best parameters
best_knn = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': np.arange(1, 50)}, cv=5)
best_knn.fit(X, y)
print("Best Params:", best_knn.best_params_)
