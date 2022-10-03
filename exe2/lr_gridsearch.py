from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from utils import *

def grid_search_params():
    parameters = {'max_iter': list(range(200, 401, 25)),
            'C': [0.01, 0.15, 0.2, 0.25, 0.4],
            'solver': ['saga'],
            'penalty': ['none', 'l2'],
    }
    clf = LogisticRegression(fit_intercept = True)
    cv = StratifiedKFold(n_splits = 3, shuffle = True)
    gridsearch = GridSearchCV(clf, parameters, n_jobs = 12, cv = cv, scoring = 'roc_auc',
                                      verbose = 2, refit = True)

    _, _, train_set, train_label = get_samples("./data/train1_icu_data.csv", "./data/train1_icu_label.csv", ret_raw = True)
    white_train_set = data_whitening(train_set)
    gridsearch.fit(white_train_set, train_label.ravel())

    print(gridsearch.best_estimator_)

if __name__ == '__main__':
    grid_search_params()

    """
    LogisticRegression(C=0.01, max_iter=350, solver='saga')
    """
