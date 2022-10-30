"""
    KNN experiment with self implemented PCA
    Author: Qianyue He 
    Date: 2022.10.30
"""

import numpy as np
from pca import PCA
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
import sys
sys.path.append("..")

from general.utils import *

# If tag == true, we use the grid search result, else: we use default setting (defined by sklearn lib)
def knn_test(tag):
    _, _, train_set, train_label = get_samples("../exe2/data/train1_icu_data.csv", "../exe2/data/train1_icu_label.csv", ret_raw = True)
    _, _, test_set, test_label = get_samples("../exe2/data/test1_icu_data.csv", "../exe2/data/test1_icu_label.csv", ret_raw = True)

    # Using all default setting with no extra processing
    if tag:
        clf = Lasso(alpha = 0.01, fit_intercept = True)
        clf.fit(train_set, train_label)
        train_output = clf.predict(train_set)
        test_output = clf.predict(test_set)
        train_pred_label = (train_output > 0.5).astype(int)
        test_pred_label = (test_output > 0.5).astype(int)
        # manual CV
        cv_score = []
        for i in range(5):
            train_data_split = np.concatenate((train_set[0:1000 * i], train_set[1000 + 1000 * i:, :]), axis = 0)
            train_label_split = np.concatenate((train_label[0:1000 * i], train_label[1000 + 1000 * i:, :]), axis = 0)
            test_data_split = train_set[1000 * i:1000 * i + 1000, :]
            test_label_split = train_label[1000 * i:1000 * i + 1000, :]
            clf = Lasso(alpha = 0.01, fit_intercept = True)
            clf.fit(train_data_split, train_label_split)
            test_output = clf.predict(test_data_split)
            cv_pred_label = (test_output > 0.5).astype(int)
            test_acc = acc_calculate(cv_pred_label, test_label_split)
            cv_score.append(test_acc)
    else:
        clf = KNeighborsClassifier()
        pca = PCA(n_components = 16, whiten = True)
        train_set = pca.transform(train_set)
        test_set = pca.transform(test_set)
        clf.fit(train_set, train_label.ravel())
        train_pred_label = clf.predict(train_set)
        test_pred_label = clf.predict(test_set)
        cv_score = cross_val_score(clf, train_set, train_label.ravel(), cv = 5)

    train_acc = acc_calculate(train_pred_label, train_label)
    test_acc = acc_calculate(test_pred_label, test_label)

    print("Train set accuracy: %f, train set error rate: %f"%(train_acc, 1 - train_acc))
    print("Test set accuracy: %f, test set error rate: %f"%(test_acc, 1 - test_acc))
    print("Cross validation: ", cv_score)

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
        knn_test(0)
