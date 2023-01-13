import numpy as np
from _01_LoadDataset.LoadingDataset import LoadDataset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class Variant_01_PCA_KMeans(object):
    def __init__(self, path, vis, logger):
        self.path = path
        self.vis = vis
        self.logger = logger
        self.dataset = LoadDataset(self.path, self.logger)
        self.dataloader, self.y_labels = self.dataset.loadData()
        self.pca_components = 2
        self.pca_transformed = []
        self.feature_extraction()
        self.clustering()

    def feature_extraction(self):
        pca = PCA(n_components=self.pca_components)
        self.pca_transformed = pca.fit_transform(self.dataloader.aligned_spikes)
        self.logger.info(f"Number of Samples after PCA: {len(self.pca_transformed)}")
        self.logger.info(f"First Spike Frame after PCA: {self.pca_transformed[0]}")

    def clustering(self):
        number_of_clusters = max(self.y_labels) + 1
        self.logger.info(f"Number of Clusters: {number_of_clusters}")

        kmeans = KMeans(n_clusters=number_of_clusters)
        kmeans.fit(self.pca_transformed)

        self.logger.info(kmeans.labels_)
        for i in range(0, number_of_clusters):
            self.logger.info(f"Cluster {i}: "
                             f"Occurrences: {(self.y_labels == i).sum()} "
                             f"KMEANS: {(kmeans.labels_ == i).sum()}")

        x = []
        y = []
        for spike in self.pca_transformed:
            x.append(spike[0])
            y.append(spike[1])

        centroids = self.vis.getClusterCenters(self.pca_transformed, kmeans.labels_)

        self.vis.visualisingClusters(x, y, kmeans.labels_, centroids)
        self.vis.printConfusionMatrix(self.y_labels, kmeans.labels_, np.unique(self.y_labels))
