import numpy as np
import sys
sys.path.append("..")
from general.utils import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 26 + 26 + 11 - 1
    train_pos, train_neg, train_set, train_label = get_samples("../exe2/data/train1_icu_data.csv", "../exe2/data/train1_icu_label.csv", ret_raw = True)
    main_feat_1 = train_pos[:, 26 + 26 + 11 - 1]
    main_feat_2 = train_neg[:, 26 + 26 + 11 - 1]
    plt.hist(main_feat_1, np.linspace(0, 1.0, 50), alpha = 0.5, label = 'positive main feature')
    plt.hist(main_feat_2, np.linspace(0, 1.0, 50), alpha = 0.5, label = 'negative main feature')
    plt.legend()
    plt.grid(axis = 'both')
    plt.show()