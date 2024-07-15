import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Churn_Modelling.csv')

print(data)

# data preprocessing
X= data.iloc[:,3:13].values
Y = data.iloc[:,13].values

# encoding
from sklearn import preprocessing
# geography
le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])
# gender
le2 = preprocessing.LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],remainder="passthrough")
X = ohe.fit_transform(X)
X = X[:,1:]

# train test
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.33, random_state = 0)

# scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)