import numpy as np
from _01_LoadDataset.LoadingDataset import LoadDataset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class Variant_01_PCA_KMeans(object):
    def __init__(self, path, vis, logger, parameter_logger, pca_components=2):
        self.path = path
        self.vis = vis
        self.logger = logger
        self.parameter_logger = parameter_logger
        self.pca_components = pca_components
        self.parameter_logger.info(f"PCA Components: {self.pca_components}")

        self.dataset = LoadDataset(self.path, self.logger)
        self.dataloader, self.y_labels = self.dataset.load_data()
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
        self.parameter_logger.info(f"K-Means Number of Clusters: {number_of_clusters}")

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

        centroids_true = self.vis.get_cluster_centers(self.pca_transformed, self.y_labels)
        centroids_kmeans = self.vis.get_cluster_centers(self.pca_transformed, kmeans.labels_)

        self.vis.visualising_features(x, y)
        self.vis.visualising_clusters(x, y, self.y_labels, centroids_true, "true")
        self.vis.visualising_clusters(x, y, kmeans.labels_, centroids_kmeans, "kmeans")
        self.vis.print_metrics(self.y_labels, kmeans.labels_)
