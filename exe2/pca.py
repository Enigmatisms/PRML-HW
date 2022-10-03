"""
    My PCA module, since sklearn can not satisfy the need.
    Author: Qianyue He 
    Date: 2022.10.1
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import data_whitening

class PCA:
    def __init__(self, n_components = 3, whiten = False):
        self.n_components = n_components
        self.whiten = whiten

        self.principal_comps = None
        self.principal_ratio = None

    def forward(self, X: np.ndarray):
        if self.whiten:
            X = data_whitening(X, True)
        corr_x = X.T @ X
        U, S, _ = np.linalg.svd(corr_x)

        s_sum = np.sum(S)
        self.principal_ratio = S[:self.n_components] / s_sum

        self.principal_comps = U[:, :self.n_components]

    def plot_principal(self, show = True):
        plt.figure(0)
        plt.imshow(np.diag(self.principal_ratio))
        plt.colorbar()

        contribution = self.principal_comps ** 2
        plt.figure(1)
        plt.imshow(contribution.T)
        plt.colorbar()

        if show:
            plt.show()
