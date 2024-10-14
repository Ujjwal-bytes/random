import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(r"C:\Users\ujjwal jha\Downloads\pima-indians-diabetes.data.csv", names=['preg','plas','pres','skin','test','mass','pedi','age','class'])
X, Y = df.iloc[:, :-1], df.iloc[:, -1]

model = AdaBoostClassifier(n_estimators=30, random_state=7)
print(cross_val_score(model, X, Y).mean())