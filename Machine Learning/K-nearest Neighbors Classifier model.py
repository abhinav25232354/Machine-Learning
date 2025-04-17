from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
# pip install scikit-learn
# pip install sklearn

iris = datasets.load_iris()
# print(iris.DESCR)

features = iris.data
labels = iris.target
# print(features[0], labels[0])

# Training Classifier
clf = KNeighborsClassifier()
clf.fit(features, labels)

# Giving Four Parameters
'''
    - sepal length in cm
    - sepal width in cm
    - petal length in cm
    - petal width in cm
'''
a = 10
b = 20
c = a + b
print(c)

preds = clf.predict([[1, 1, 1, 1]])
'''
    - Iris-Setosa [0]
    - Iris-Versicolour [1]
    - Iris-Virginica [2]
'''
print(preds)


''' if classifier returns 0 then it means the flower is iris setosa'''