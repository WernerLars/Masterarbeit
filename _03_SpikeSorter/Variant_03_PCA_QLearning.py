import numpy as np

from _01_LoadDataset.LoadingDataset import LoadDataset
from sklearn.decomposition import PCA
from _02_Classes_Autoencoder_QLearning.QLearning import Q_Learning


class Variant_03_PCA_QLearning(object):
    def __init__(self, path, vis, logger, parameter_logger,
                 pca_components=2, q_learning_size=300):
        self.path = path
        self.vis = vis
        self.logger = logger
        self.parameter_logger = parameter_logger
        self.pca_components = pca_components
        self.parameter_logger.info(f"PCA Components: {self.pca_components}")
        self.q_learning_size = q_learning_size
        self.parameter_logger.info(f"Q Learning Size: {q_learning_size}")

        self.dataset = LoadDataset(self.path, self.logger)
        self.dataloader, self.y_labels = self.dataset.loadData()
        self.pca_transformed = []

        self.feature_extraction()
        self.clustering()

    def feature_extraction(self):
        pca = PCA(n_components=self.pca_components)
        self.pca_transformed = pca.fit_transform(self.dataloader.aligned_spikes)
        self.logger.info(f"Number of Samples after PCA: {len(self.pca_transformed)}")
        self.logger.info(f"First Spike Frame after PCA: {self.pca_transformed[0]}")

    def clustering(self):
        ql = Q_Learning(self.parameter_logger, self.pca_components)
        for s in range(0, 2):
            ql.addToFeatureSet(self.pca_transformed[s])

        self.logger.info(f"Number of Spikes for Q_Learning: {self.q_learning_size}")
        for s in range(0, self.q_learning_size):
            ql.dynaQAlgorithm(self.pca_transformed[s])
            self.logger.info(f"Q_Learning: {s:>5d}/{self.q_learning_size:>5d}]")
            print(f"Q_Learning: {s:>5d}/{self.q_learning_size:>5d}]")
        x = []
        y = []
        for spike in ql.spikes:
            x.append(spike[0])
            y.append(spike[1])

        self.logger.info(ql.clusters)
        self.logger.info(ql.randomFeatures)

        centroids_true = self.vis.getClusterCenters(ql.spikes,
                                                    self.y_labels[:self.q_learning_size])
        centroids_qlearning = self.vis.getClusterCenters(ql.spikes, ql.clusters)

        self.vis.visualisingFeatures(x, y)
        self.vis.visualisingClusters(x, y, self.y_labels[:self.q_learning_size]
                                     , centroids_true, "true")
        self.vis.visualisingClusters(x, y, ql.clusters, centroids_qlearning, "qlearning")

        self.vis.printMetrics(self.y_labels[:self.q_learning_size],
                              ql.clusters,
                              np.unique(self.y_labels[:self.q_learning_size]))
