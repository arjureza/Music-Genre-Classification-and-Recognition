# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:16:02 2021

@author: Devan
"""

#Importing the necessary packages and libaries
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import numpy as np


iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state = 0)

linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train, y_train)


linear_pred = linear.predict(X_test)
poly_pred = poly.predict(X_test)

accuracy_lin = linear.score(X_test, y_test)
accuracy_poly = poly.score(X_test, y_test)

print("Misclassification Linear Kernel:", 1 - accuracy_lin)
print("Misclassification Polynomial Kernel:", 1 - accuracy_poly)