import pandas as pd
import numpy as np
from missingpy import MissForest

df = pd.read_csv("Updated.csv")
y = df['default_ind']
df1 = df.drop(['application_key', 'default_ind'], axis=1)

imputer = MissForest()
df2 = imputer.fit_transform(df1)

np.savetxt("miss_forest.csv", df2, delimiter=",")
