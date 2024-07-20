import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

from randforest import RandomForest

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = RandomForest(n_trees=3)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(clf.accuracy(y_test, y_pred))




