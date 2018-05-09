# coding=utf-8
import pprint

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from queue import Queue


def gini(labels):
    """
    计算基尼系数
    :param labels: 标签集合
    :return: 该集合的基尼系数
    """
    _, a = np.unique(labels, return_counts=True)
    return 1 - np.sum((a / np.sum(a)) ** 2)


def discrete_f(x, y):
    """
    计算根据某离散特征划分后的基尼系数
    :param x: 待划分样本的单一特征
    :param y: 待划分样本的标签
    :return: gini系数和特征可取的值
    """
    gini_val = 0
    n_sample = x.shape[0]
    x_t, x_c = np.unique(x, return_counts=True)
    for t, c in zip(x_t, x_c):
        gini_val += c / float(n_sample) * gini(y[x == t])
    return gini_val, x_t


def continuous_f(x, y):
    """
    为连续特征计算gini系数
    :param x: 待划分样本的单一特征
    :param y: 待划分样本的标签
    :return: 最优的gini系数和划分值
    """
    # 连续特征
    assert x.shape == y.shape
    x_t = np.unique(x)
    if len(x_t) == 1:
        # 特征都相同，取频率最大的
        pass
    x_t.sort()
    sep_vals = (x_t[:-1] + x_t[1:]) / 2
    min_gini_val = 1
    best_sep_val = None
    for sep_val in sep_vals:
        tmp_gini = gini(y[x <= sep_val]) + gini(y[x > sep_val])
        tmp_gini /= 2
        if min_gini_val > tmp_gini:
            min_gini_val = tmp_gini
            best_sep_val = sep_val
    # if best_sep_val is None:
    #     print(best_sep_val)
    return min_gini_val, best_sep_val


def find_best_feature(X_, y_, sample_mask, feature_mask, feature_types):
    """
    寻找划分的最优特征
    :param X_: 所有的训练样本
    :param y_: 所有的训练标签
    :param sample_mask: 样本掩码
    :param feature_mask: 特征掩码
    :param feature_types:
    :return: 最优划分特征的下标和用于划分的值
    """
    X, y = X_[sample_mask], y_[sample_mask]
    # print(X.shape, y.shape)
    n_sample, n_col = X.shape
    min_gini = 1
    best_index = None
    best_sep_val = []
    shape = X.shape
    for i in range(n_col):
        if not feature_mask[i]:
            continue  # 如果该特征已经用过了，就跳过
        if feature_types[i] == "c":
            gini_val, sep_val = continuous_f(X[:, i], y)
        else:
            gini_val, sep_val = discrete_f(X[:, i], y)
        if min_gini > gini_val:
            best_index = i
            best_sep_val = sep_val
    return best_index, best_sep_val


def build_tree(X, y,
               sample_mask,
               feature_mask,
               feature_types,
               depth,
               max_depth):
    """
    递归构建决策树的过程
    :param X: 总训练样本
    :param y: 总训练标签
    :param sample_mask: 当前节点剩余的样本的掩码，起到下标的作用，避免数据拷贝
    :param feature_mask: 特征的掩码，表明哪些特征还可以用于划分
    :param feature_types: 每个特征对应的数据类型
    :param depth: 当前节点的深度
    :param max_depth: 最大深度
    :return: 构建好的决策树节点
    """
    result = dict()
    _y = y[sample_mask]
    # 检查y是否都是同一类
    y_t, y_c = np.unique(_y, return_counts=True)
    if len(y_t) == 1:
        result["is_leaf"] = True
        result["class"] = y_t[0]
        return result
    # 样本类别不同，则检查深度
    # print(depth)
    ind = np.argmax(y_c)
    most_common_y = y[ind]
    result["most_common_type"] = most_common_y
    if depth == max_depth:
        result["is_leaf"] = True
        result["class"] = most_common_y
        return result
    # 深度没有到达上限，则继续分类
    result["is_leaf"] = False
    best_index, sep_val = find_best_feature(X, y, sample_mask, feature_mask, feature_types)
    new_feature_mask = np.copy(feature_mask)

    if feature_types[best_index] == "c":
        # 连续型特征
        result["feature_type"] = "c"
        result["index"] = best_index
        result["sep_val"] = sep_val
        new_sample_mask = (X[:, best_index] <= sep_val) & sample_mask
        reverse_new_sample_mask = (X[:, best_index] > sep_val) & sample_mask
        sons = dict()
        sons["le"] = build_tree(
            X, y,
            new_sample_mask,
            new_feature_mask,
            feature_types, depth + 1,
            max_depth)
        sons["gt"] = build_tree(
            X, y,
            reverse_new_sample_mask,
            new_feature_mask,
            feature_types,
            depth + 1,
            max_depth)
        result["sons"] = sons
    else:
        # 离散型特征
        result["feature_type"] = "d"
        result["index"] = best_index
        new_feature_mask[best_index] = 0  # 只有离散型的特征只用一次，将feature_mask相应位置0
        sons = dict()
        for t in sep_val:
            new_sample_mask = (X[:, best_index] == t) & sample_mask
            sons[t] = build_tree(
                X, y,
                new_sample_mask,
                new_feature_mask,
                feature_types, depth + 1,
                max_depth)
        result["sons"] = sons
    return result


def get_feature_types(X):
    """
    确定特征的类型，是连续还是离散
    :param X: 样本数据
    :return: 与特征维数相同的一组向量，对应位置记录该处的特征是连续c，还是离散d
    """
    n_sample, n_col = X.shape
    feature_types = []
    for i in range(n_col):
        feature_types.append("c")

    return np.array(feature_types)


def decide_class(x, tree_node):
    """
    根据学习得到的决策树对样本进行分类
    :param x: 单个样本
    :param tree_node: 决策树的某个节点
    :return: 单个样本的预测分类
    """
    if tree_node['is_leaf']:
        return tree_node["class"]
    feature_type = tree_node["feature_type"]
    idx = tree_node["index"]
    if feature_type == "c":
        sep_val = tree_node["sep_val"]
        if x[idx] <= sep_val:
            return decide_class(x, tree_node["sons"]["le"])
        else:
            return decide_class(x, tree_node["sons"]["gt"])
    else:

        if x[idx] in tree_node["sons"]:
            return decide_class(x, tree_node["sons"][x[idx]])
        else:
            return tree_node["most_common_type"]


def prunning_decide_class(x, y, tree_node):
    """
    剪枝的时候将验证集存储到叶节点上
    :param x: 
    :param y: 
    :param tree_node: 
    :return: 
    """
    if tree_node['is_leaf']:
        if "valid" not in tree_node:
            tree_node["valid"] = []
        tree_node["valid"].append(y)
        return tree_node["class"]
    feature_type = tree_node["feature_type"]
    idx = tree_node["index"]
    if feature_type == "c":
        sep_val = tree_node["sep_val"]
        if x[idx] <= sep_val:
            return prunning_decide_class(x, y, tree_node["sons"]["le"])
        else:
            return prunning_decide_class(x, y, tree_node["sons"]["gt"])
    else:
        return prunning_decide_class(x, y, tree_node["sons"][x[idx]])


def pruning(X, y, tree_):
    """
    剪枝
    :param X: 
    :param y: 
    :param tree_: 
    :return: 
    """
    print("starting prunning")
    # 先将验证集的结果推到叶节点
    while True:
        pruning_num = 0
        n_samples, n_col = X.shape
        for i in range(n_samples):
            prunning_decide_class(X[i], y[i], tree_)
        # 找到所有子节点都是叶子的节点
        q = []
        q.append(tree_)
        while q:
            tree_node = q.pop(0)
            if not tree_node["is_leaf"]:
                flag = True
                most_common_y = tree_node["most_common_type"]
                old_acc = 0
                new_acc = 0
                valid_list = []
                for son in tree_node["sons"].values():
                    if not son["is_leaf"]:
                        q.append(son)
                        flag = False
                    else:
                        if "valid" in son:
                            valid_list += son["valid"]
                            old_acc += np.sum(np.array(son["valid"]) == son["class"])
                            new_acc += np.sum(np.array(son["valid"]) == most_common_y)
                if flag:
                    if new_acc > old_acc:
                        print("pruning {}".format(tree_node))
                        pruning_num += 1
                        tree_node["is_leaf"] = True
                        tree_node['class'] = most_common_y
                        tree_node["valid"] = valid_list
        if not pruning_num:
            break


class DecisionTreeClassifer(object):
    """
    决策树分类器
    """

    def __init__(self, max_depth=100):
        self.max_depth = 100
        self.tree_ = None

    def fit(self, X, y, is_pruning=False):
        if is_pruning:
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5, shuffle=True)
        else:
            X_train, y_train = X, y
        feature_types = get_feature_types(X_train)
        n_samples, n_col = X_train.shape
        # 训练
        self.tree_ = build_tree(
            X_train, y_train,
            sample_mask=np.ones(n_samples, dtype=bool),
            feature_mask=np.ones(n_col),
            feature_types=feature_types,
            depth=0,
            max_depth=self.max_depth
        )
        # 剪枝
        if is_pruning:
            pruning(X_valid, y_valid, self.tree_)

    def predict(self, X_test):
        if not self.tree_:
            raise Exception("not trained yet")
        n_samples, n_col = X_test.shape
        pred_y = np.zeros(n_samples, dtype="int64")
        for i in range(n_samples):
            pred_y[i] = decide_class(X_test[i], self.tree_)
        return pred_y

    def score(self, X, y):
        pred_y = self.predict(X)
        n_samples, n_col = X.shape
        return np.sum(y == pred_y) / n_samples


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    classif = DecisionTreeClassifer()
    classif.fit(X_train, y_train)
    # pprint.pprint(classif.tree_)
    pruning(X_test, y_test, classif.tree_)
    # print(classif.score(X_train, y_train))
    # print(classif.score(X_test, y_test))
