import pandas as pd
from missingpy import MissForest

df = pd.read_csv("testX_Updated.csv")
df1 = df.drop(['application_key'], axis=1)

imputer = MissForest()
df2 = imputer.fit_transform(df1)

df2.to_csv("testX_forest.csv")

df2.to_csv("Impute_Forest.csv")
