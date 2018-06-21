# coding=utf-8
import pprint

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


def mse(labels):
    mu = np.mean(labels)
    return np.sum((labels - mu) ** 2)


def find_best_sep(x, y):
    assert x.shape == y.shape

    x_t, x_c = np.unique(x, return_counts=True)

    if len(x_t) == 1:
        return x_t[0], mse(y)

    x_t.sort()

    best_mse = mse(y[x <= x_t[0]]) + mse(y[x > x_t[0]])
    best_sep = x_t[0]

    for i in x_t[1:-1]:
        mse_val = mse(y[x <= i]) + mse(y[x > i])
        sep_val = i
        if best_mse > mse_val:
            best_mse = mse_val
            best_sep = sep_val

    return best_sep, best_mse


def find_best_feature(X_, y_, sample_mask):
    """
    寻找划分的最优特征
    :param X_: 所有的训练样本
    :param y_: 所有的训练标签
    :param sample_mask: 样本掩码
    :return: 最优划分特征的下标和用于划分的值
    """
    X, y = X_[sample_mask], y_[sample_mask]
    n_sample, n_col = X.shape

    best_index = 0
    best_sep_val, min_mse = find_best_sep(X[:, 0], y)

    for i in range(1, n_col):
        # if i == 11:
        #     print(i)
        sep_val, mse_val = find_best_sep(X[:, i], y)
        if min_mse > mse_val:
            min_mse = mse_val
            best_index = i
            best_sep_val = sep_val
    # print(min_mse, best_index)
    return best_index, best_sep_val


def build_tree(X, y,
               sample_mask,
               depth,
               max_depth):
    """
    递归构建决策树的过程
    :param X: 总训练样本
    :param y: 总训练标签
    :param sample_mask: 当前节点剩余的样本的掩码，起到下标的作用，避免数据拷贝
    :param depth: 当前节点的深度
    :param max_depth: 最大深度
    :return: 构建好的决策树节点
    """
    result = dict()
    _y = y[sample_mask]

    # if _y.shape[0] == 42:
    #     print(_y.shape)

    # 检查y的mse是否为0
    if len(_y) == 1 or mse(_y) == 0:
        result["is_leaf"] = True
        result["val"] = _y[0]
        return result

    # 如果设置了max_depth
    if max_depth and depth == max_depth:
        result["is_leaf"] = True
        result["val"] = np.mean(_y)
        return result

    # 深度没有到达上限，则继续分类
    result["is_leaf"] = False
    best_index, sep_val = find_best_feature(X, y, sample_mask)

    result["index"] = best_index
    result["val"] = sep_val
    new_sample_mask = (X[:, best_index] <= sep_val) & sample_mask
    reverse_new_sample_mask = (X[:, best_index] > sep_val) & sample_mask

    # print((new_sample_mask > 0).sum(), (reverse_new_sample_mask > 0).sum())

    result["le"] = build_tree(
        X, y,
        new_sample_mask,
        depth + 1,
        max_depth)
    result["gt"] = build_tree(
        X, y,
        reverse_new_sample_mask,
        depth + 1,
        max_depth)

    return result


def decide_val(x, tree_node):
    """
    根据学习得到的决策树对样本进行分类
    :param x: 单个样本
    :param tree_node: 决策树的某个节点
    :return: 单个样本的预测分类
    """
    if tree_node['is_leaf']:
        return tree_node["val"]
    idx = tree_node["index"]
    sep_val = tree_node["val"]
    if x[idx] <= sep_val:
        return decide_val(x, tree_node["le"])
    else:
        return decide_val(x, tree_node["gt"])


class DecisionTreeRegressor(object):
    """
    决策树回归
    """

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree_ = None

    def fit(self, X, y):
        n_samples, n_col = X.shape
        assert n_samples == len(y)
        self.tree_ = build_tree(
            X, y,
            sample_mask=np.ones(n_samples, dtype=bool),
            depth=0,
            max_depth=self.max_depth
        )

    def predict(self, X):
        if not self.tree_:
            raise Exception("not trained yet")
        n_samples, n_col = X.shape
        pred_y = np.zeros(n_samples, dtype="float64")
        for i in range(n_samples):
            pred_y[i] = decide_val(X[i], self.tree_)
        return pred_y

    def score(self, X, y):
        pred_y = self.predict(X)
        # return r2_score(y, pred_y)
        return mean_squared_error(y, pred_y)


if __name__ == "__main__":
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    classif = DecisionTreeRegressor()
    classif.fit(X_train, y_train)
    # pprint.pprint(classif.tree_)
    # print(classif.score(X_train, y_train))
    print(classif.score(X_test, y_test))
