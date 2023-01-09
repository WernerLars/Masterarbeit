import numpy as np

from _01_LoadDataset.LoadingDataset import LoadDataset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def Variant_01_PCA_KMeans(path, vis, logger):
    dataset = LoadDataset()
    dataloader, y_labels = dataset.loadData(path, logger)

    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(dataloader.aligned_spikes)
    logger.info(f"Number of Samples after PCA: {len(pca_transformed)}")
    logger.info(f"First Spike Frame after PCA: {pca_transformed[0]}")

    number_of_clusters = max(y_labels) + 1
    logger.info(f"Number of Clusters: {number_of_clusters}")

    kmeans = KMeans(n_clusters=number_of_clusters)
    kmeans.fit(pca_transformed)

    logger.info(kmeans.labels_)
    for i in range(0, number_of_clusters):
        logger.info(f"Cluster {i}: Occurrences: {(y_labels == i).sum()}  KMEANS: {(kmeans.labels_ == i).sum()}")

    x = []
    y = []
    for spike in pca_transformed:
        x.append(spike[0])
        y.append(spike[1])

    centroids = vis.getClusterCenters(pca_transformed, kmeans.labels_)

    vis.visualisingClusters(x, y, kmeans.labels_, centroids)
    vis.printConfusionMatrix(y_labels, kmeans.labels_, np.unique(y_labels), logger)
