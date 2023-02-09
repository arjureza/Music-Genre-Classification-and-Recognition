# K-Means Clustering using sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
import numpy as np

#pylint: disable=no-member
# iris = load_iris()
# X = iris.data
# y = iris.target

#pylint: disable=no-member
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)

print(X[1, :])
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

k_means = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300)
k_means.fit(X_train)
prediction = k_means.predict(X_test)

# k_means = KMeans(n_clusters=4, init='random', max_iter=300, n_init=10, random_state=0)
# k_means.fit(X_train)
# prediction = k_means.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
print(1 - accuracy)