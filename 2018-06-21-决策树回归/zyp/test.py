# coding=utf-8
from sklearn.datasets import load_boston, load_diabetes
# from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
import cart
import sklearn.tree
import numpy as np

# X, y = load_boston(return_X_y=True)
X, y = load_diabetes(return_X_y=True)

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,shuffle=True)

sklearn_model = sklearn.tree.DecisionTreeRegressor()
my_model = cart.DecisionTreeRegressor()

sklearn_res=[]
my_res=[]

for train_index,test_index in KFold(5,shuffle=True).split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    sklearn_model.fit(X_train,y_train)
    my_model.fit(X_train,y_train)

    sklearn_res.append(r2_score(sklearn_model.predict(X_test),y_test))
    my_res.append(r2_score(my_model.predict(X_test),y_test))

print np.array(sklearn_res).mean()
print np.array(my_res).mean()
