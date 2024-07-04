import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score

data = pd.read_csv('salaries.csv')

x = data.iloc[:,2:5]
y = data.iloc[:,5:]

X = x.values
Y = y.values

# string cannot be converted to float so, did not take the 2nd column
print(data.drop(data.columns[1], axis=1).corr())
# measures the relationship between two variables.

#linear regression
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X,Y)


model = sm.OLS(linReg.predict(X),X)
print(model.fit().summary())

print('Linear R² value = ' + str(r2_score(Y, linReg.predict(X))))