import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('tenis.csv')

# converting categorical data to numerical data with using encoder
from sklearn.preprocessing import LabelEncoder
data2 = data.apply(LabelEncoder().fit_transform)

c = data2.iloc[:,:1]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
c=ohe.fit_transform(c).toarray()
print(c)

weather = pd.DataFrame(data = c, index = range(14), columns=['o','r','s'])
lastData = pd.concat([weather,data.iloc[:,1:3]],axis = 1)
lastData = pd.concat([data2.iloc[:,-2:],lastData], axis = 1)


# splitting data for training and testing
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(lastData.iloc[:,:-1],lastData.iloc[:,-1:],test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)

print(y_pred)

import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=lastData.iloc[:,:-1], axis=1 )
X_l = lastData.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = lastData.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

lastData = lastData.iloc[:,1:]

import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=lastData.iloc[:,:-1], axis=1 )
X_l = lastData.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog = lastData.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)







