import pandas as pd
import numpy as np
from typing import Tuple, Union

def reader(path: str):
    head_row = pd.read_csv(path, nrows = 0)
    csv_data = pd.read_csv(path, usecols = list(head_row))
    return csv_data.to_numpy()

def data_whitening(data: np.ndarray, only_unbias = False) -> np.ndarray:
    # input data should be of shape (n_samples, n_features)
    mean = np.mean(data, axis = 0)  # mean for features
    if only_unbias:
        return data - mean
    std = np.std(data, axis = 0)
    std = np.maximum(np.full_like(std, 1e-5), std)
    return (data - mean) / std

# return (class1, class2)
def get_samples(feats_file: str, label_file: str, ret_raw = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    features = reader(feats_file)
    labels = reader(label_file)
    flag = (labels > 0).ravel()
    if not ret_raw:
        return features[flag], features[~flag]
    return features[flag], features[~flag], features, labels

def acc_calculate(pred: np.ndarray, target: np.ndarray) -> float :
    pred = pred.ravel()
    target = target.ravel()
    return np.sum(pred == target) / target.shape[0]

if __name__ == '__main__':
    X, Y = get_samples("./data/train1_icu_data.csv", "./data/train1_icu_label.csv")
    print(X.shape, Y.shape)
    