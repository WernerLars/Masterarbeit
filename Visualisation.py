import matplotlib.pyplot as plt
import numpy as np


def visualisingClusters(x, y, cluster, centroids):
    plt.figure(figsize=(8, 8))
    unique_cluster = np.unique(cluster)
    for i in unique_cluster:
        x_i = []
        y_i = []
        for j in range(0, len(cluster)):
            if i == cluster[j]:
                x_i.append(x[j])
                y_i.append(y[j])
        plt.scatter(x_i, y_i)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k', marker='+')
    plt.show()
    return


def visualisingReconstructedSpike(original, reconstructed, n_features, cluster):
    plt.figure(figsize=(4, 4))
    t = np.arange(0, n_features, 1)
    title = "True Label: "+cluster
    plt.plot(t, original, label="original")
    plt.plot(t, reconstructed, label="reconstructed")
    plt.legend(loc="upper left")
    plt.title(title)
    plt.show()
    return


#visualisingClusters([1.0771875, 0.30116683, 0.80967987], [1.1234825, 0.28021073, 0.9399729])
#visualisingReconstructedSpike([1, 2], [1, 3], 2)
