from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import sys
sys.path.append("..")
from general.utils import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # grid_search_params()
    _, _, train_set, train_label = get_samples("../exe2/data/train1_icu_data.csv", "../exe2/data/train1_icu_label.csv", ret_raw = True)
    _, _, test_set, test_label = get_samples("../exe2/data/test1_icu_data.csv", "../exe2/data/test1_icu_label.csv", ret_raw = True)

    clf = RandomForestClassifier(criterion = 'entropy', max_depth = 8, n_estimators = 90, bootstrap = False, oob_score = False)
    clf.fit(train_set, train_label.ravel())
    train_pred_label = clf.predict(train_set)
    test_pred_label = clf.predict(test_set)
    cv_score = cross_val_score(clf, train_set, train_label.ravel(), cv = 10)

    train_acc = acc_calculate(train_pred_label, train_label)
    test_acc = acc_calculate(test_pred_label, test_label)

    print("Train set accuracy: %f, train set error rate: %f"%(train_acc, 1 - train_acc))
    print("Test set accuracy: %f, test set error rate: %f"%(test_acc, 1 - test_acc))
    # print("OOB scores", clf.oob_score_)
    print("Cross validation: ", cv_score, " average: ", np.mean(cv_score))
    # print()

    importance = clf.feature_importances_
    with open('../exe2/data/train1_icu_data.csv', 'r') as file:
        line = file.readline()[:-1]
        line = line.split(',')
    all_items = list(zip(importance, line))
    all_items.sort(key = lambda x: x[0], reverse = True)

    plt.bar([item[1] for item in all_items[:20]], [item[0] for item in all_items[:20]])
    plt.xticks(rotation = 75)
    print(all_items)
    plt.show()