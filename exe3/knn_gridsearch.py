"""
    KNN experiment
    Author: Qianyue He 
    Date: 2022.10.4
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import sys
import tqdm
sys.path.append("..")

from general.utils import *

# If tag == true, we use the grid search result, else: we use default setting (defined by sklearn lib)
def knn_test(tag):
    _, _, train_set, train_label = get_samples("../exe2/data/train1_icu_data.csv", "../exe2/data/train1_icu_label.csv", ret_raw = True)
    _, _, test_set, test_label = get_samples("../exe2/data/test1_icu_data.csv", "../exe2/data/test1_icu_label.csv", ret_raw = True)

    # Using all default setting with no extra processing
    if tag:
        clf = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=22, weights='distance')
    else:
        clf = KNeighborsClassifier()
    # pca = PCA(n_components = 16)
    # train_set = pca.fit_transform(train_set, train_label)
    # test_set = pca.fit_transform(test_set, test_label)

    clf.fit(train_set, train_label.ravel())
    train_pred_label = clf.predict(train_set)
    test_pred_label = clf.predict(test_set)

    train_acc = acc_calculate(train_pred_label, train_label)
    test_acc = acc_calculate(test_pred_label, test_label)
    cv_score = cross_val_score(clf, train_set, train_label.ravel(), cv = 5)

    print("Train set accuracy: %f, train set error rate: %f"%(train_acc, 1 - train_acc))
    print("Test set accuracy: %f, test set error rate: %f"%(test_acc, 1 - test_acc))
    print("Cross validation: ", cv_score)
    print("NN num: %d"%(clf.n_neighbors))

    """
        Default param
        Train set accuracy: 0.770400, train set error rate: 0.229600
        Test set accuracy: 0.672744, test set error rate: 0.327256
        NN num: 5
    """

    """
        Default param with PCA (n_components = 16)
        Train set accuracy: 0.770400, train set error rate: 0.229600
        Test set accuracy: 0.672744, test set error rate: 0.327256
        NN num: 5
    """

def k_influence():
    _, _, train_set, train_label = get_samples("../exe2/data/train1_icu_data.csv", "../exe2/data/train1_icu_label.csv", ret_raw = True)
    _, _, test_set, test_label = get_samples("../exe2/data/test1_icu_data.csv", "../exe2/data/test1_icu_label.csv", ret_raw = True)
    train_accs = []
    test_accs = []
    for i in tqdm.tqdm(range(3, 50)):
        clf = KNeighborsClassifier(n_neighbors = i, algorithm = 'auto')
        clf.fit(train_set, train_label.ravel())
        train_pred_label = clf.predict(train_set)
        test_pred_label = clf.predict(test_set)
        train_acc = acc_calculate(train_pred_label, train_label)
        test_acc = acc_calculate(test_pred_label, test_label)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
    xs = np.arange(len(train_accs))

    plt.scatter(xs, train_accs, c = 'r', s = 8)
    plt.scatter(xs, test_accs, c = 'b', s = 8)

    plt.plot(xs, train_accs, label = 'training acc', c = 'r')
    plt.plot(xs, test_accs, label = 'testing acc', c = 'b')

    plt.xlabel('nearest neighbor number')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(axis = 'both')
    plt.show()


def knn_grid_search():
    parameters = {'weights': ['uniform', 'distance'],
            'algorithm' : ['ball_tree', 'kd_tree', 'auto'],
            'n_neighbors': list(range(3, 40)),
    }
    clf = KNeighborsClassifier()

    _, _, train_set, train_label = get_samples("../exe2/data/train1_icu_data.csv", "../exe2/data/train1_icu_label.csv", ret_raw = True)
    # pca = PCA(n_components = 10)
    # train_set = pca.fit_transform(train_set, train_label)


    cv = StratifiedKFold(n_splits = 4, shuffle = True)
    gridsearch = GridSearchCV(clf, parameters, n_jobs = 12, cv = cv, scoring = 'roc_auc',
                                      verbose = 2, refit = True)
    gridsearch.fit(train_set, train_label.ravel())
    print(gridsearch.best_estimator_)

    """
        GridSearch only on train set
        KNeighborsClassifier(algorithm='ball_tree', n_neighbors=34, weights='distance')
        Train set accuracy: 1.000000, train set error rate: 0.000000
        Test set accuracy: 0.693710, test set error rate: 0.306290
    """

if __name__ == "__main__":
    from sys import argv
    if len(argv) > 1:
        try:
            tag = int(argv[1])
        except ValueError:
            print("So stupid. You want to convert '%s' to int?"%(argv[1]))
            exit(1)
        knn_test(tag > 0)
    else:
        knn_grid_search()
    k_influence()
