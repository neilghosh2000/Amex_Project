import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics

df = pd.read_csv("Updated.csv")
y = df['default_ind']
df1 = df.drop(['application_key', 'default_ind'], axis=1)

df2 = df1.fillna(df1.mean())

X_train, X_test, y_train, y_test = train_test_split(df2, y, test_size=0.4,
                                                    random_state=1)
# create logistic regression object
reg = linear_model.LogisticRegression()

# train the model using the training sets
reg.fit(X_train, y_train)

# making predictions on the testing set
y_pred = reg.predict(X_test)

# comparing actual response values (y_test) with predicted response values (y_pred)
print("Logistic Regression model accuracy(in %):",
      metrics.accuracy_score(y_test, y_pred) * 100)