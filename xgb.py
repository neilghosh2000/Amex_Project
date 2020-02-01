import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv("Updated.csv")
df1 = df.dropna(thresh=30)
y = df1['default_ind']
df2 = df1.drop(['application_key', 'default_ind'], axis=1)

df3 = df2.fillna(df2.mean())
X_train, X_test, y_train, y_test = train_test_split(df3, y)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))