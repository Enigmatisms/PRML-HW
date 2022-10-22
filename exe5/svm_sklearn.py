"""
    KNN experiment
    Author: Qianyue He 
    Date: 2022.10.4
"""

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score
import sys
sys.path.append("..")
from general.utils import *


def grid_search_params():
    parameters = {'C': [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto']
    }
    clf = SVC(kernel = 'poly')
    scaler = StandardScaler()

    _, _, train_set, train_label = get_samples("../exe2/data/train1_icu_data.csv", "../exe2/data/train1_icu_label.csv", ret_raw = True)
    train_set = scaler.fit_transform(train_set)

    cv = StratifiedKFold(n_splits = 5, shuffle = True)
    gridsearch = GridSearchCV(clf, parameters, n_jobs = 12, cv = cv, scoring = 'roc_auc',
                                      verbose = 2, refit = True)
    gridsearch.fit(train_set, train_label.ravel())
    print(gridsearch.best_estimator_)

    """
        Best params for linear SVM: SVC(C=0.1, degree=2, kernel='linear')
        Best params for RBF SVM: SVC(degree=2)
        Best params for Poly SVM: SVC(C=0.5, gamma='auto', kernel='poly', degree=3)
    """

def svm_test():
    scaler = StandardScaler()

    parameters = {'C': 0.5,
            'kernel' : 'poly',
            'degree': 3,
            'gamma': 'auto'
    }

    clf = SVC(**parameters)

    _, _, train_set, train_label = get_samples("../exe2/data/train1_icu_data.csv", "../exe2/data/train1_icu_label.csv", ret_raw = True)
    _, _, test_set, test_label = get_samples("../exe2/data/test1_icu_data.csv", "../exe2/data/test1_icu_label.csv", ret_raw = True)

    train_set = scaler.fit_transform(train_set)
    test_set = scaler.fit_transform(test_set)

    clf.fit(train_set, train_label)
    cv_score = cross_val_score(clf, train_set, train_label.ravel(), cv = 5)
    train_pred = clf.predict(train_set)
    test_pred = clf.predict(test_set)
    train_set_acc = acc_calculate(train_pred, train_label)
    test_set_acc = acc_calculate(test_pred, test_label)

    print("Train set accuracy: %f, train set error rate: %f"%(train_set_acc, 1 - train_set_acc))
    print("Test set accuracy: %f, test set error rate: %f"%(test_set_acc, 1 - test_set_acc))
    print("Cross validation score: ", cv_score)

if __name__ == "__main__":
    svm_test()

    """
        Linear SVM:
        Train set accuracy: 0.800200, train set error rate: 0.199800
        Test set accuracy: 0.790337, test set error rate: 0.209663
        Cross validation score:  [0.785 0.781 0.813 0.792 0.766]

        RBF SVM:
        Train set accuracy: 0.878400, train set error rate: 0.121600
        Test set accuracy: 0.787603, test set error rate: 0.212397
        Cross validation score:  [0.778 0.775 0.821 0.788 0.78 ]

        Poly SVM:
        Train set accuracy: 0.864400, train set error rate: 0.135600
        Test set accuracy: 0.781222, test set error rate: 0.218778
        Cross validation score:  [0.764 0.772 0.794 0.764 0.763]
    """