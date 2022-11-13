from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV
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
    
if __name__ == "__main__":
    grid_search_params()