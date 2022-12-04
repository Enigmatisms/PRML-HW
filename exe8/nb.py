from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score

import sys
sys.path.append("..")
from general.utils import *

def grid_search_params():
    parameters = {'var_smoothing': [10**(-i) for i in range(5, 10)] + [1e-7 + 2e-8 * i for i in (-3, -2, -1, 1, 2, 3)]}
    clf = GaussianNB()
    cv = StratifiedKFold(n_splits = 10, shuffle = True)
    gridsearch = GridSearchCV(clf, parameters, n_jobs = 16, cv = cv, scoring = 'roc_auc',
                                      verbose = 2, refit = True)

    _, _, train_set, train_label = get_samples("../exe2/data/train1_icu_data.csv", "../exe2/data/train1_icu_label.csv", ret_raw = True)
    gridsearch.fit(train_set, train_label.ravel())

    print(gridsearch.best_params_)

    """
    Best param: {'var_smoothing': 1.3999999999999998e-07}
    """

def pred_with_loss(clf: GaussianNB, data: np.ndarray, gt_label: np.ndarray, FP_risk: float = 0.5, FN_risk: float = 1.0):
    proba = clf.predict_proba(data)

    risk_1 = (proba[:, 0] * FP_risk)
    risk_0 = (proba[:, 1] * FN_risk)

    result = np.zeros(proba.shape[0], dtype = np.int64)
    result[risk_1 < risk_0] = 1
    return result

def naive_bayes_training():
    clf = GaussianNB(var_smoothing = 1.4e-7)
    _, _, train_set, train_label = get_samples("../exe2/data/train1_icu_data.csv", "../exe2/data/train1_icu_label.csv", ret_raw = True)
    _, _, test_set, test_label = get_samples("../exe2/data/test1_icu_data.csv", "../exe2/data/test1_icu_label.csv", ret_raw = True)

    clf.fit(train_set, train_label.ravel())
    train_pred_label = clf.predict(train_set)
    test_pred_label = clf.predict(test_set)
    cv_score = cross_val_score(clf, train_set, train_label.ravel(), cv = 10)

    train_acc = acc_calculate(train_pred_label, train_label)
    test_acc = acc_calculate(test_pred_label, test_label)

    get_risk = lambda labels, FP_risk = 0.5, FN_risk = 1.0: np.sum(labels * FP_risk + FN_risk * (1 - labels))

    train_risk_before = get_risk(train_pred_label)
    test_risk_before = get_risk(test_pred_label)
    

    print("Train set accuracy: %f, train set error rate: %f"%(train_acc, 1 - train_acc))
    print("Test set accuracy: %f, test set error rate: %f"%(test_acc, 1 - test_acc))
    print("Cross validation: ", cv_score, " average: ", np.mean(cv_score))
    print("Train risk: ", train_risk_before, ", test risk: ", test_risk_before)

    train_pred_label = pred_with_loss(clf, train_set, train_label)
    test_pred_label = pred_with_loss(clf, test_set, test_label)

    train_risk_after = get_risk(train_pred_label)
    test_risk_after = get_risk(test_pred_label)

    train_acc = acc_calculate(train_pred_label, train_label)
    test_acc = acc_calculate(test_pred_label, test_label)

    print("Train set result: %f, train set error rate: %f"%(train_acc, 1 - train_acc))
    print("Test set accuracy: %f, test set error rate: %f"%(test_acc, 1 - test_acc))
    print("Train risk: ", train_risk_after, ", test risk: ", test_risk_after)

if __name__ == "__main__":
    naive_bayes_training()