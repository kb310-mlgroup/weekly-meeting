import pprint

import time
from sklearn import tree
from sklearn.datasets import load_iris, load_digits,load_wine,load_breast_cancer
from sklearn.model_selection import train_test_split
from dt import DecisionTreeClassifer

X,y =load_iris(return_X_y=True)
# X, y = load_digits(return_X_y=True)
# X, y = load_wine(return_X_y=True)
# X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

cl1 = tree.DecisionTreeClassifier()
cl1.fit(X_train, y_train)

print(cl1.score(X_train, y_train))
print(cl1.score(X_test, y_test))

cl2 = DecisionTreeClassifer()
cl2.fit(X_train, y_train)

# pprint.pprint(cl2.tree_)

print(cl2.score(X_train, y_train))

print(cl2.score(X_test, y_test))
