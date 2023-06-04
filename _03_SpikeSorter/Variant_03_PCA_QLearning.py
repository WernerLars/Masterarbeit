from tqdm import tqdm

from _01_LoadDataset.LoadingDataset import LoadDataset
from sklearn.decomposition import PCA
from _02_Classes_Autoencoder_QLearning.QLearning import Q_Learning


class Variant_03_PCA_QLearning(object):
    def __init__(self, path, vis, logger, parameter_logger, normalise=False,
                 pca_components=2, q_learning_size=None,
                 punishment_coefficient=0.6,
                 disable_tqdm=False):
        self.path = path
        self.vis = vis
        self.logger = logger
        self.parameter_logger = parameter_logger
        self.pca_components = pca_components
        self.normalise = normalise
        self.parameter_logger.info(f"PCA Components: {self.pca_components}")
        self.q_learning_size = q_learning_size
        self.parameter_logger.info(f"Q Learning Size: {q_learning_size}")
        self.logger.info(f"Punishment_Coefficient: {punishment_coefficient}")
        self.disable_tqdm = disable_tqdm

        self.dataset = LoadDataset(self.path, self.logger)
        self.dataloader, self.y_labels = self.dataset.load_data()
        self.pca_transformed = []

        self.ql = Q_Learning(self.parameter_logger, self.pca_components,
                             normalise=self.normalise,
                             punishment_coefficient=punishment_coefficient)

        self.feature_extraction()
        self.clustering()

    def feature_extraction(self):
        pca = PCA(n_components=self.pca_components)
        self.pca_transformed = pca.fit_transform(self.dataloader.aligned_spikes)
        self.logger.info(f"Number of Samples after PCA: {len(self.pca_transformed)}")
        self.logger.info(f"First Spike Frame after PCA: {self.pca_transformed[0]}")

    def clustering(self):
        if self.normalise:
            for s in range(0, 2):
                self.ql.add_to_feature_set(self.pca_transformed[s])

        x = []
        y = []
        self.logger.info(f"Number of Spikes for Q_Learning: {self.q_learning_size}")
        if self.q_learning_size is None:
            self.q_learning_size = len(self.pca_transformed)

        q_learning_loop = tqdm(enumerate(self.pca_transformed[:self.q_learning_size]), total=self.q_learning_size,
                               disable=self.disable_tqdm)
        q_learning_loop.set_description(f"Q_Learning")
        for s, features in q_learning_loop:
            self.ql.dyna_q_algorithm(features)
            x.append(features[0])
            y.append(features[1])

        self.logger.info(self.ql.clusters)
        self.logger.info(self.ql.randomFeatures)
        self.ql.print_q_table()
        self.ql.print_model()

        centroids_true = self.vis.get_cluster_centers(self.pca_transformed,
                                                      self.y_labels[:self.q_learning_size])
        centroids_qlearning = self.vis.get_cluster_centers(self.pca_transformed, self.ql.clusters)

        self.vis.compute_cluster_distances(self.pca_transformed, self.y_labels[:self.q_learning_size], centroids_true)

        self.vis.visualising_features(x, y)
        self.vis.visualising_clusters(x, y, self.y_labels[:self.q_learning_size], centroids_true, "true")
        self.vis.visualising_clusters(x, y, self.ql.clusters, centroids_qlearning, "qlearning")

        self.vis.print_metrics(self.y_labels[:self.q_learning_size], self.ql.clusters)
