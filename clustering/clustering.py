import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('customers.csv')

X = data.iloc[:,3:].values

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
YGuess = ac.fit_predict(X)
print(YGuess)

plt.scatter(X[YGuess==0,0],X[YGuess==0,1],s=100, c='red')
plt.scatter(X[YGuess==1,0],X[YGuess==1,1],s=100, c='blue')
plt.scatter(X[YGuess==2,0],X[YGuess==2,1],s=100, c='green')
plt.scatter(X[YGuess==3,0],X[YGuess==3,1],s=100, c='yellow')
plt.show()

# dendogram

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.show()