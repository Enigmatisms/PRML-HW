from sklearn.linear_model import Perceptron
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from utils import *

def grid_search_params():
    parameters = {'max_iter': list(range(20, 201, 10)),
            'l1_ratio': [0.0, 0.05, 0.15, 0.25],
            'penalty': [None, 'l2','l1','elasticnet'],
            'alpha': [0.0001, 0.01, 0.]
    }
    clf = Perceptron(fit_intercept = True, shuffle = True)
    cv = StratifiedKFold(n_splits = 3, shuffle = True)
    gridsearch = GridSearchCV(clf, parameters, n_jobs = 12, cv = cv, scoring = 'roc_auc',
                                      verbose = 2, refit = True)

    _, _, train_set, train_label = get_samples("./data/train1_icu_data.csv", "./data/train1_icu_label.csv", ret_raw = True)
    gridsearch.fit(train_set, train_label.ravel())

    print(gridsearch.best_estimator_)

    # Best estimator: Perceptron(alpha=0.01, l1_ratio=0.0, max_iter=40, penalty='l1')


if __name__ == '__main__':
    grid_search_params()
