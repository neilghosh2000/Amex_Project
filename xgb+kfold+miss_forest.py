import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


df = pd.read_csv("miss_forest.csv")
y = df['default_ind']
df1 = df.drop(['application_key', 'default_ind'], axis=1)

# df2 = df1.fillna(df1.mean())
model = XGBClassifier()

X = np.array(df1)

kf = KFold(n_splits=15)
KFold(n_splits=15, random_state=None, shuffle=True)
for train_index, test_index in kf.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

