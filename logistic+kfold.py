import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Updated.csv")
y = df['default_ind']
df1 = df.drop(['application_key', 'default_ind'], axis=1)

df2 = df1.fillna(df1.mean())

# create logistic regression object
reg = linear_model.LogisticRegression()

# Initialise the Scaler
scaler = StandardScaler()

# To scale data
scaler.fit(df2)

X = np.array(df2)
kf = KFold(n_splits=10)
KFold(n_splits=10, random_state=None, shuffle=True)
for train_index, test_index in kf.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred)*100)