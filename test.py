import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

df = pd.read_csv("Updated.csv")
df1 = df.dropna(thresh=39)
y = df1['default_ind']
df2 = df1.drop(['application_key', 'default_ind'], axis=1)

testdf = pd.read_csv("testX.csv")
x_testx = testdf.drop(['application_key'], axis=1)

df3 = df2.fillna(df2.mean())
model = XGBClassifier()

max_acc = 0
X_final = []
Y_final = []

X = np.array(df3)
z = np.array(y)
kf = KFold(n_splits=15)
KFold(n_splits=15, random_state=None, shuffle=True)
for train_index, test_index in kf.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = z[train_index], z[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)

    if max_acc < accuracy:
        max_acc = accuracy
        X_final = X_train
        Y_final = y_train

    print("Accuracy: %.2f%%" % (accuracy * 100.0))

model.fit(X_final, Y_final)
y_pred = model.predict(x_testx)
