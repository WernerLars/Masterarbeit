import numpy as np

from _01_LoadDataset.LoadingDataset import LoadDataset
from sklearn.decomposition import PCA
from _02_Classes_Autoencoder_QLearning.QLearning import Q_Learning


class Variant_03_PCA_QLearning(object):
    def __init__(self, path, vis, logger, parameter_logger,
                 pca_components=2, q_learning_size=None,
                 punishment_coefficient=1):
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

        self.ql = Q_Learning(self.parameter_logger, self.pca_components, punishment_coefficient=punishment_coefficient)

        self.feature_extraction()
        self.clustering()

    def feature_extraction(self):
        pca = PCA(n_components=self.pca_components)
        self.pca_transformed = pca.fit_transform(self.dataloader.aligned_spikes)
        self.logger.info(f"Number of Samples after PCA: {len(self.pca_transformed)}")
        self.logger.info(f"First Spike Frame after PCA: {self.pca_transformed[0]}")

    def clustering(self):
        #for s in range(0, 2):
        #    self.ql.addToFeatureSet(self.pca_transformed[s])

        x = []
        y = []
        self.logger.info(f"Number of Spikes for Q_Learning: {self.q_learning_size}")
        if self.q_learning_size is None:
            self.q_learning_size = len(self.pca_transformed)

        for s in range(0, self.q_learning_size):
            features = self.pca_transformed[s]
            self.ql.dynaQAlgorithm(features)
            self.logger.info(f"Q_Learning: {s:>5d}/{self.q_learning_size:>5d}]")
            print(f"Q_Learning: {s:>5d}/{self.q_learning_size:>5d}]")
            x.append(features[0])
            y.append(features[1])

        self.logger.info(self.ql.clusters)
        self.logger.info(self.ql.randomFeatures)

        centroids_true = self.vis.getClusterCenters(self.pca_transformed,
                                                    self.y_labels[:self.q_learning_size])
        centroids_qlearning = self.vis.getClusterCenters(self.pca_transformed, self.ql.clusters)

        self.vis.visualisingFeatures(x, y)
        self.vis.visualisingClusters(x, y, self.y_labels[:self.q_learning_size]
                                     , centroids_true, "true")
        self.vis.visualisingClusters(x, y, self.ql.clusters, centroids_qlearning, "qlearning")

        self.vis.printMetrics(self.y_labels[:self.q_learning_size],
                              self.ql.clusters,
                              np.unique(self.y_labels[:self.q_learning_size]))
