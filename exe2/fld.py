"""
    Fisher Linear Discriminant Implementation
    Author: Qianyue He (Enigmatisms)
    Date: 2022.9.30
"""

import numpy as np
import matplotlib.pyplot as plt

class FLD:
    def __init__(self):
        self.m1 = None
        self.m2 = None
        self.w = None
        self.n1 = 0
        self.n2 = 0
        self.w0 = 0.

    @staticmethod
    def single_class_scatter(data: np.ndarray, mean: np.ndarray) -> np.ndarray:
        data_unbiased = data - mean.T
        return data_unbiased.T @ data_unbiased

    def get_hyperplane(self, c1_data: np.ndarray, c2_data: np.ndarray) -> np.ndarray:
        # input must be of shape (n_samples, n_features)
        # I am afraid that in FLD, we can not do data whitening since it removes mean values
        self.m1 = c1_data.mean(axis = 0).reshape(-1, 1)
        self.m2 = c2_data.mean(axis = 0).reshape(-1, 1)
        self.n1, self.n2 = c1_data.shape[0], c2_data.shape[0]
        Sw = FLD.single_class_scatter(c1_data, self.m1) + FLD.single_class_scatter(c2_data, self.m2)
        # In case Sw is big (matrix inversion is O(n^3), so is SVD yet svd is generally faster)
        if Sw.shape[0] > 6:
            u, s, _ = np.linalg.svd(Sw)
            s[abs(s) < 1e-5] = 1e-5                     # getting rid of the numerical instability
            inv_Sw = u @ np.diag(1. / s) @ u.T
        else:
            inv_Sw = np.linalg.inv(Sw)
        result_w = inv_Sw @ (self.m1 - self.m2)
        result_w /= np.linalg.norm(result_w)
        self.w = result_w / np.linalg.norm(result_w)      # normalize
        self.w0 = self.w.T @ (self.n1 * self.m1 + self.n2 * self.m2) / (self.n1 + self.n2)
        return self.w

    # in order to separate different classes, we need to add class_bias (of shape (1, n_features))
    @staticmethod
    def generate_testing_samples(n_samples: int, n_features: int, class_bias: np.ndarray) -> np.ndarray:
        result = np.random.normal(0, 1, (n_samples, n_features))
        result *= np.abs(np.random.normal(0, 2, (1, n_features)))   # random variance
        result += np.random.normal(0, 10, (1, n_features))          # random bias   
        return result + class_bias

    @staticmethod
    def plot_two_classes(c1_data: np.ndarray, c2_data: np.ndarray, show = True):
        plt.scatter(c1_data[:, 0], c1_data[:, 1], color = 'red', alpha = 0.7, label = 'class 1')
        plt.scatter(c2_data[:, 0], c2_data[:, 1], color = 'blue', alpha = 0.7, label = 'class 2')
        plt.legend()
        if show:
            plt.show()

    # input (samples is of shape [n_features, n_samples]), output (1, classes - True means class 0, False means class 1)
    def predict_class(self, samples: np.ndarray, transpose = False) -> np.ndarray:
        if transpose:
            samples = samples.T
        if self.w is None:
            raise RuntimeError("You did not feed the training data. How dare you use the predictor? You want me to guess?")
        mult = self.w.T @ samples
        return mult > self.w0

    # only for two dim data
    def plot_hyperplane(self):
        if self.w.shape[0] > 2:
            raise RuntimeError("Data higher than 2d is difficult or impossible to visualize.")
        intersect_pt = (self.n1 * self.m1 + self.n2 * self.m2) / (self.n1 + self.n2)
        direction = np.array([[-self.w[1, 0]], [self.w[0, 0]]])
        xs = np.linspace(-8, 8, 3)
        pts = intersect_pt + direction * xs
        print(intersect_pt.shape, direction.shape, xs.shape)
        plt.plot(pts[0, :], pts[1, :], c = 'k', alpha = 0.7, label = 'hyperplane')

# unit test for FLD
def visualization_test():
    # homo特有的随机数臭种子（悲）
    np.random.seed(114514 + 1)

    fld = FLD()
    class_1 = FLD.generate_testing_samples(30, 2, np.array([[4., 12.]]))
    class_2 = FLD.generate_testing_samples(30, 2, np.array([[4., -8.]]))
    plt.figure(0)
    FLD.plot_two_classes(class_1, class_2, False)
    fld.get_hyperplane(class_1, class_2)

    all_class_samples = np.concatenate((class_1.T, class_2.T), axis = -1)
    result = fld.predict_class(all_class_samples)
    fld.plot_hyperplane()
    print(result, np.sum(result))
    plt.grid(axis = 'both')
    plt.show()

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from general.utils import *
    
    fld = FLD()
    train_pos, train_neg, raw_set, labels = get_samples("./data/train1_icu_data.csv", "./data/train1_icu_label.csv", ret_raw = True)
    _, _, raw_test, test_labels = get_samples("./data/test1_icu_data.csv", "./data/test1_icu_label.csv", ret_raw = True)

    fld.get_hyperplane(train_pos, train_neg)
    train_pred_label = fld.predict_class(raw_set, True)
    test_pred_label = fld.predict_class(raw_test, True)

    train_set_acc = acc_calculate(train_pred_label, labels)
    test_set_acc = acc_calculate(test_pred_label, test_labels)

    print("Train set accuracy: %f, train set error rate: %f"%(train_set_acc, 1 - train_set_acc))
    print("Test set accuracy: %f, test set error rate: %f"%(test_set_acc, 1 - test_set_acc))
    # visualization_test()


    
