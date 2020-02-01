import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

pca = PCA(n_components=40)

df = pd.read_csv("Updated.csv")
df1 = df.dropna(thresh=39)
y = df1['default_ind']
df2 = df1.drop(['application_key', 'default_ind'], axis=1)

df3 = df2.fillna(df2.mean())
model = XGBClassifier()

X = np.array(df3)
z = np.array(y)
kf = KFold(n_splits=15)
KFold(n_splits=15, random_state=None, shuffle=True)
for train_index, test_index in kf.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = z[train_index], z[test_index]
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy: %.2f%%" % (accuracy * 100.0))

