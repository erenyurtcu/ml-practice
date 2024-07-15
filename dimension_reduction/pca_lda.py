import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Wine.csv')
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values

# train test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PCA 
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

# Logistic Regression before PCA transformation
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Logistic Regression after PCA transformation
classifier2 = LogisticRegression(random_state = 0)
classifier2.fit(X_train2, y_train)

# predictions
y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix

# actual / results without PCA
print('actual / without PCA')
cm = confusion_matrix(y_test, y_pred)
print(cm)

# actual / results after PCA
print("actual / with PCA")
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

# PCA results / results without PCA
print('without PCA and with PCA')
cm3 = confusion_matrix(y_pred, y_pred2)
print(cm3)

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components = 2)

X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# After LDA transformation
classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda, y_train)

# Predict using LDA data
y_pred_lda = classifier_lda.predict(X_test_lda)

# LDA results / original results
print('LDA and original')
cm4 = confusion_matrix(y_pred, y_pred_lda)
print(cm4)

