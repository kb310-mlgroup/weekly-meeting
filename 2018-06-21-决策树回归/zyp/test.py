# coding=utf-8
from sklearn.datasets import load_boston, load_diabetes
# from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import cart
import sklearn.tree

X, y = load_boston(return_X_y=True)
# X, y = load_diabetes(return_X_y=True)

sklearn_model = sklearn.tree.DecisionTreeRegressor()
my_model = cart.DecisionTreeRegressor()


print(cross_val_score(sklearn_model,X,y,cv=5).mean())
print(cross_val_score(sklearn_model,X,y,cv=5).mean())