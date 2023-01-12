import numpy as np
from _01_LoadDataset.LoadingDataset import LoadDataset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class Variant_01_PCA_KMeans(object):
    def __init__(self, path, vis, logger):
        self.path = path
        self.vis = vis
        self.logger = logger
        self.dataset = LoadDataset()
        self.dataloader, self.y_labels = self.dataset.loadData(self.path, self.logger)
        self.train()

    def train(self):
        pca = PCA(n_components=2)
        pca_transformed = pca.fit_transform(self.dataloader.aligned_spikes)
        self.logger.info(f"Number of Samples after PCA: {len(pca_transformed)}")
        self.logger.info(f"First Spike Frame after PCA: {pca_transformed[0]}")

        number_of_clusters = max(self.y_labels) + 1
        self.logger.info(f"Number of Clusters: {number_of_clusters}")

        kmeans = KMeans(n_clusters=number_of_clusters)
        kmeans.fit(pca_transformed)

        self.logger.info(kmeans.labels_)
        for i in range(0, number_of_clusters):
            self.logger.info(f"Cluster {i}: Occurrences: {(self.y_labels == i).sum()}  KMEANS: {(kmeans.labels_ == i).sum()}")

        x = []
        y = []
        for spike in pca_transformed:
            x.append(spike[0])
            y.append(spike[1])

        centroids = self.vis.getClusterCenters(pca_transformed, kmeans.labels_)

        self.vis.visualisingClusters(x, y, kmeans.labels_, centroids)
        self.vis.printConfusionMatrix(self.y_labels, kmeans.labels_, np.unique(self.y_labels), self.logger)
