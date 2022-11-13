import numpy as np
import sys
sys.path.append("..")
from general.utils import *
import matplotlib.pyplot as plt

MAIN_1 = (26 + 26 + 11 - 1, np.linspace(0, 1, 50))
MAIN_2 = (22, [0, 0.25, 0.5, 0.75, 1])
MAIN_3 = (9, np.linspace(4, 127, 50))
MAIN_4 = (12, np.arange(0, 8))
MAIN_5 = (10, np.linspace(0, 12, 50))

def histogram(main_id, bins: np.ndarray):
    train_pos, train_neg, _, _ = get_samples("../exe2/data/train1_icu_data.csv", "../exe2/data/train1_icu_label.csv", ret_raw = True)
    main_feat_1 = train_pos[:, main_id]
    main_feat_2 = train_neg[:, main_id]
    print(main_feat_1.max(), main_feat_2.max(), main_feat_1.min(), main_feat_2.min())
    plt.hist(main_feat_1, bins, alpha = 0.5, label = 'positive main feature')
    plt.hist(main_feat_2, bins, alpha = 0.5, label = 'negative main feature')
    plt.legend()
    plt.grid(axis = 'both')
    plt.show()

if __name__ == "__main__":
    # 26 + 26 + 11 - 1
    histogram(*MAIN_5)