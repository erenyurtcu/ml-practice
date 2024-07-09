import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('customers.csv')

X = data.iloc[:,3:].values

from sklearn.cluster import KMeans

kmeans = KMeans ( n_clusters = 3, init = 'k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)
results = []
for i in range(1,11):
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 123)
    kmeans.fit(X)
    results.append(kmeans.inertia_)

plt.scatter(range(1, 11), results, color='red')
plt.plot(range(1,11),results, color = 'blue')