import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from autoimpute.imputations import MultipleImputer
from autoimpute.analysis import MiLinearRegression


df = pd.read_csv("Updated.csv")
y = df['default_ind']
df1 = df.drop(['application_key', 'default_ind'], axis=1)
X = np.array(df1)
# simple example using default instance of MultipleImputer
imp = MultipleImputer()

# fit transform returns a generator by default, calculating each imputation method lazily
imp.fit_transform(df1)

# By default, use statsmodels OLS and MultipleImputer()
simple_lm = MiLinearRegression()

# model = XGBClassifier()

kf = KFold(n_splits=10)
KFold(n_splits=10, random_state=None, shuffle=True)
for train_index, test_index in kf.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # fit the model on each multiply imputed dataset and pool parameters
    simple_lm.fit(X_train, y_train)
    y_pred = simple_lm.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

