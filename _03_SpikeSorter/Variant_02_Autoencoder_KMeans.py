import numpy as np
from _01_LoadDataset.LoadingDataset import LoadDataset
from _01_LoadDataset.SpikeClassToPytorchDataset import SpikeClassToPytorchDataset
from _02_Classes_Autoencoder_QLearning.Autoencoder import *
from torch.utils.data import DataLoader
import torch
from torch import nn
from sklearn.cluster import KMeans


class Variant_02_Autoencoder_KMeans(object):
    def __init__(self, path, vis, logger, parameter_logger,
                 chooseAutoencoder=1, split_ratio=1, epochs=8, batch_size=1, seed=0,
                 number_of_features=2):
        self.path = path
        self.vis = vis
        self.logger = logger
        self.parameter_logger = parameter_logger
        self.chooseAutoencoder = chooseAutoencoder
        self.split_ratio = split_ratio
        self.parameter_logger.info(f"Split Ratio: {self.split_ratio}")
        self.epochs = epochs
        self.parameter_logger.info(f"Epochs: {self.epochs}")
        self.batch_size = batch_size
        self.parameter_logger.info(f"Batch Size: {self.batch_size}")
        self.seed = seed
        torch.manual_seed(self.seed)
        self.parameter_logger.info(f"Seed: {self.seed}")

        self.dataset = LoadDataset(self.path, self.logger)
        self.data, self.y_labels = self.dataset.loadData()
        self.input_size = len(self.data.aligned_spikes[0])
        self.number_of_features = number_of_features
        self.parameter_logger.info(f"Input Size: {self.input_size}")

        self.autoencoder_models = {
            1: ["Autoencoder", Autoencoder(self.input_size, self.number_of_features)],
            2: ["Convolutional Autoencoder", ConvolutionalAutoencoder(self.input_size,
                                                                      self.number_of_features)]
        }
        self.autoencoder = self.autoencoder_models[self.chooseAutoencoder][1]
        self.parameter_logger.info(f"Chosen Model: {self.autoencoder_models[self.chooseAutoencoder][0]}")
        self.parameter_logger.info(self.autoencoder)

        self.loss_function = nn.MSELoss()
        self.parameter_logger.info(self.loss_function)

        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        self.parameter_logger.info(self.optimizer)

        self.loss_values = []
        self.epoch_loss = []

        self.preprocessing()

    def preprocessing(self):
        data = SpikeClassToPytorchDataset(self.data.aligned_spikes, self.y_labels)
        dataloader = DataLoader(data, batch_size=self.batch_size)
        self.logger.info(dataloader)

        for t in range(self.epochs):
            self.logger.info(f"Epoch {t + 1}\n-------------------------------")
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train(dataloader)
            self.vis.printLossCurve(self.epoch_loss, t+1)

        self.vis.printLossCurve(self.loss_values)
        self.clustering(dataloader, self.y_labels)
        self.logger.info("Done!")

    def train(self, train_dataloader):
        size = len(train_dataloader.dataset)
        self.autoencoder.train()
        self.epoch_loss = []
        for batch, (X, y) in enumerate(train_dataloader):

            # Compute reconstruction error
            reconstructed_spike, encoded_features = self.autoencoder(X)
            loss = self.loss_function(reconstructed_spike, X)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss = loss.item()
                self.epoch_loss.append(loss)
                current = batch * len(X)
                self.logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        self.loss_values.append(sum(self.epoch_loss)/len(train_dataloader))

    def clustering(self, dataloader, y_test):
        encoded_features_list = []
        encoded_features_X = []
        encoded_features_Y = []
        cluster_labels = []

        number_of_clusters = max(y_test) + 1
        self.logger.info(f"Number of Clusters: {number_of_clusters}")
        self.parameter_logger.info(f"K-Means Number of Clusters: {number_of_clusters}")

        visualise = []
        for k in range(0, number_of_clusters):
            visualise.append(True)

        for _, (X, y) in enumerate(dataloader):
            reconstructed_spike, encoded_features = self.autoencoder(X)

            with torch.no_grad():
                x_np = X.numpy().flatten()
                encoded_features_np = encoded_features.numpy()[0]
                reconstructed_spike_np = reconstructed_spike.numpy().flatten()
                cluster = y.numpy()[0]

                encoded_features_list.append(encoded_features_np)
                encoded_features_X.append(encoded_features_np[0])
                encoded_features_Y.append(encoded_features_np[1])
                cluster_labels.append(cluster)

                if visualise[cluster]:
                    self.vis.visualisingReconstructedSpike(x_np,
                                                           reconstructed_spike_np,
                                                           len(x_np),
                                                           str(cluster))
                    self.vis.printSpike(x_np, len(x_np),
                                        "b", f"real_spike{cluster}")
                    self.vis.printSpike(reconstructed_spike_np, len(reconstructed_spike_np),
                                        "r", f"reconstructed_spike{cluster}")
                    visualise[cluster] = False

        self.logger.info(f"Number of Samples after Autoencoder testing: {len(encoded_features_list)}")
        self.logger.info(f"First Spike after testing: {encoded_features_list[0]}")

        kmeans = KMeans(n_clusters=number_of_clusters)
        kmeans.fit(encoded_features_list)

        self.logger.info(y_test)
        self.logger.info(kmeans.labels_)
        for n in range(0, number_of_clusters):
            self.logger.info(f"Cluster {n} Occurrences: {(y_test == n).sum()}; "
                             f"KMEANS: {(kmeans.labels_ == n).sum()}")

        centroids_true = self.vis.getClusterCenters(encoded_features_list, cluster_labels)
        centroids_kmeans = self.vis.getClusterCenters(encoded_features_list, kmeans.labels_)

        self.vis.visualisingFeatures(encoded_features_X, encoded_features_Y)
        self.vis.visualisingClusters(encoded_features_X, encoded_features_Y,
                                     cluster_labels, centroids_true, "true")
        self.vis.visualisingClusters(encoded_features_X, encoded_features_Y,
                                     kmeans.labels_, centroids_kmeans, "kmeans")

        self.vis.printMetrics(cluster_labels, kmeans.labels_, np.unique(cluster_labels))
