import numpy as np


def get_tree_depths(clf):
    tree_depths = [estimator.tree_.max_depth for estimator in clf.estimators_]
    return tree_depths


def get_leaves_count(clf):
    n_leaves = np.zeros(clf.n_estimators, dtype=int)
    for i in range(clf.n_estimators):
        n_nodes = clf.estimators_[i].tree_.node_count
        # use left or right children as you want
        children_left = clf.estimators_[i].tree_.children_left
        for x in range(n_nodes):
            if children_left[x] == -1:
                n_leaves[i] += 1
    return n_leaves