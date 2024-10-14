import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from warnings import filterwarnings
filterwarnings("ignore")

df = pd.read_csv(r"C:\Users\ujjwal jha\Downloads\ai-perform.csv").apply(lambda x: x.astype('category').cat.codes if x.dtype == 'object' else x)
X, y = df.drop('Play Tennis', axis=1), df['Play Tennis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB().fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)), "Accuracy:", accuracy_score(y_test, model.predict(X_test)))