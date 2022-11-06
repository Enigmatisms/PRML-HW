from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import sys
sys.path.append("..")
from general.utils import *
import matplotlib.pyplot as plt

def grid_search_params():
    parameters = {'n_estimators': list(range(50, 91, 5)),
            'criterion' : ["gini", "entropy", "log_loss"],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'bootstrap': [True, False],
            'oob_score': [True, False]
    }
    clf = RandomForestClassifier()
    cv = StratifiedKFold(n_splits = 10, shuffle = True)
    gridsearch = GridSearchCV(clf, parameters, n_jobs = 16, cv = cv, scoring = 'roc_auc',
                                      verbose = 2, refit = True)

    _, _, train_set, train_label = get_samples("../exe2/data/train1_icu_data.csv", "../exe2/data/train1_icu_label.csv", ret_raw = True)
    gridsearch.fit(train_set, train_label.ravel())

    print(gridsearch.best_params_)

    # Best estimator for dataset 1: {'criterion': 'entropy', 'max_depth': 28, 'n_estimators': 79}
    # Best estimator for dataset 2: {'criterion': 'entropy', 'max_depth': 28, 'n_estimators': 79}


if __name__ == '__main__':
    grid_search_params()
    # _, _, train_set, train_label = get_samples("../exe2/data/train1_icu_data.csv", "../exe2/data/train1_icu_label.csv", ret_raw = True)
    # clf = RandomForestClassifier(criterion = 'entropy', max_depth = 28, n_estimators = 79)
    # clf.fit(train_set, train_label.ravel())
    # # print()

    # importance = clf.feature_importances_
    # with open('../exe2/data/train1_icu_data.csv', 'r') as file:
    #     line = file.readline()[:-1]
    #     line = line.split(',')
    # all_items = list(zip(importance, line))
    # all_items.sort(key = lambda x: x[0], reverse = True)

    # plt.bar([item[1] for item in all_items[:20]], [item[0] for item in all_items[:20]])
    # plt.xticks(rotation = 75)
    # print(all_items)
    # plt.show()