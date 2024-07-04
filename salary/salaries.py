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

# linear regression
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X,Y)

model = sm.OLS(linReg.predict(X),X)
print(model.fit().summary())
print('Linear R² value = ' + str(r2_score(Y, linReg.predict(X))))

# polynomial regression

from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree = 4)
xPoly = polyReg.fit_transform(X)
print(xPoly)
linReg2 = LinearRegression()
linReg2.fit(xPoly,y)

for i in range(X.shape[1]):
    sorted_indices = np.argsort(X[:, i])
    X_sorted = X[sorted_indices]
    Y_sorted = Y[sorted_indices]
    
    plt.figure(figsize=(10,2))
    plt.scatter(X_sorted[:, i], Y_sorted, color='red', label='Data points')  
    plt.plot(X_sorted[:, i], linReg2.predict(polyReg.fit_transform(X_sorted)), color='green', label='Polynomial fit')
    plt.xlabel(f'Feature {i+1}')
    plt.ylabel('Target')
    plt.title(f'Polynomial Regression (Degree 4) for Feature {i+1}')
    plt.legend()
    plt.show()
    
    
# guesses
model2 = sm.OLS(linReg2.predict(polyReg.fit_transform(X)),X)
print('poly OLS \n' + str(model2.fit().summary()))

print('Polynomial R² value = ' + str(r2_score(Y, linReg2.predict(polyReg.fit_transform(X)))))


# scaling datas

from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()
xScaled = sc1.fit_transform(X)
sc2=StandardScaler()
yScaled = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))


from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(xScaled,yScaled)


print('SVR OLS')
model3=sm.OLS(svr_reg.predict(xScaled),xScaled)
print(model3.fit().summary())

print('SVR R² value = ' + str(r2_score(yScaled, svr_reg.predict(xScaled))))
