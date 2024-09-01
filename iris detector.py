from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20, metric='minkowski',p=2)
x = iris.data
y = iris.target
knn.fit(x, y)
#put your own target in
xx = np.array([[5, 3, 6, 2]])
yy = knn.predict(xx)
xx = np.array([[5, 3, 6, 2]])
yy = knn.predict(xx)
if yy==np.array([0]):
    print("setosa")
elif yy==np.array([1]):
    print("versicolor")
elif yy==np.array([2]):
    print("virginica")