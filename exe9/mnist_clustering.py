from mlxtend.data import loadlocal_mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def get_mnist(path:str):
    return loadlocal_mnist(
            images_path=f'{path}train-images.idx3-ubyte', 
            labels_path=f'{path}train-labels.idx1-ubyte')
    
def multi_imshow(X: np.ndarray, rows = 4, cols = 4):
    for i in range(rows):
        for j in range(cols):
            img_id = cols * i + j
            plt.subplot(rows, cols, 1 + img_id)
            plt.imshow(X[img_id].reshape(28, 28))
            plt.axis('off')
    plt.show()
    
def vanilla_kmeans_clustering(data: np.ndarray, n_cluster = 4, num_examples = 4):
    clt = KMeans(n_clusters = n_cluster)
    labels = clt.fit_predict(data)
    results = []
    for i in range(n_cluster):
        examples = data[labels == i][:num_examples]
        results.append(examples)
    return np.concatenate(results, axis = 1).reshape(n_cluster * num_examples, -1)

def pca_kmeans_clustering(data: np.ndarray, n_cluster = 4, num_examples = 4, scree_plot = False):
    pca = PCA(n_components = 100)
    if scree_plot:
        pca.fit(data)
        scree_values = pca.explained_variance_ratio_
        xs = np.arange(len(scree_values))
        line_xs = np.arange(100, dtype = np.float32)
        line_ys = np.full_like(line_xs, 0.0146)
        plt.plot(xs, scree_values, c = 'r')
        plt.scatter(xs, scree_values, c = 'r', s = 8)
        plt.plot(line_xs, line_ys, c = 'b', alpha = 0.7, linestyle = '--')
        plt.grid(axis = 'both')
        plt.show()
    else:
        clt = KMeans(n_clusters = n_cluster)
        transformed = pca.fit_transform(data)
        labels = clt.fit_predict(transformed)
        results = []
        for i in range(n_cluster):
            examples = data[labels == i][:num_examples]
            results.append(examples)
        return np.concatenate(results, axis = 1).reshape(n_cluster * num_examples, -1)
    
if __name__ == "__main__":
    num_clusters = 2
    examples = 6
    use_pca = True
    X, y = get_mnist("../exe2/data/")
    is_three = y == 7
    images = X[is_three]
    if use_pca:
        clustered = pca_kmeans_clustering(images, num_clusters, examples, scree_plot = True)
    else:
        clustered = vanilla_kmeans_clustering(images, num_clusters, examples)
    if clustered is not None:
        multi_imshow(clustered, examples, num_clusters)
    