import numpy as np
from _01_LoadDataset.LoadingDataset import LoadDataset
from _01_LoadDataset.SpikeClassToPytorchDataset import SpikeClassToPytorchDataset
from _02_Classes_Autoencoder_QLearning.Autoencoder import *
from torch.utils.data import DataLoader
import torch
from torch import nn
from sklearn.cluster import KMeans


class Variant_02_Autoencoder_KMeans(object):
    def __init__(self, path, vis, logger):
        self.path = path
        self.vis = vis
        self.logger = logger
        self.dataset = LoadDataset(self.path, self.logger)
        self.data, self.y_labels = self.dataset.loadData()
        self.split_ratio = 0.8
        self.input_size = len(self.data.aligned_spikes[0])
        self.autoencoder_models = {
            1: Autoencoder(self.input_size),
            2: ConvolutionalAutoencoder(self.input_size)
        }
        self.autoencoder = self.autoencoder_models[1]
        self.logger.info(self.autoencoder)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        self.epochs = 8
        self.batch_size = 1
        self.preprocessing()

    def preprocessing(self):
        torch.manual_seed(0)
        self.logger.info(f"Input Size: {self.input_size}")

        train_idx = round(len(self.data.aligned_spikes) * self.split_ratio)
        self.logger.info(f"Train Index: {train_idx}")

        x_train = self.data.aligned_spikes[0:train_idx]
        y_train = self.y_labels[0:train_idx]
        x_test = self.data.aligned_spikes[train_idx:]
        y_test = self.y_labels[train_idx:]

        self.logger.info(f"x_train: {len(x_train)}")
        self.logger.info(f"y_train: {len(y_train)}")
        self.logger.info(f"x_test: {len(x_test)}")
        self.logger.info(f"y_test: {len(y_test)}")

        train_d = SpikeClassToPytorchDataset(x_train, y_train)
        train_dl = DataLoader(train_d, batch_size=self.batch_size)
        self.logger.info(train_dl)
        test_d = SpikeClassToPytorchDataset(x_test, y_test)
        test_dl = DataLoader(test_d, batch_size=self.batch_size)
        self.logger.info(test_dl)

        for t in range(self.epochs):
            self.logger.info(f"Epoch {t + 1}\n-------------------------------")
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train(train_dl)

        self.test(test_dl, y_test)
        self.logger.info("Done!")

    def train(self, train_dataloader):
        size = len(train_dataloader.dataset)
        self.autoencoder.train()
        for batch, (X, y) in enumerate(train_dataloader):

            # Compute prediction error
            reconstructed_spike, encoded_features = self.autoencoder(X)
            loss = self.loss_function(reconstructed_spike, X)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                self.logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self, dataloader, y_test):
        encoded_features_list = []
        encoded_features_X = []
        encoded_features_Y = []
        y_l = []

        number_of_clusters = max(y_test) + 1
        self.logger.info(f"Number of Clusters: {number_of_clusters}")

        visualise = []
        for k in range(0, number_of_clusters):
            visualise.append(True)

        for batch, (X, y) in enumerate(dataloader):
            reconstructed_spike, encoded_features = self.autoencoder(X)

            with torch.no_grad():
                encoded_features_list.append(encoded_features.numpy()[0])
                encoded_features_X.append(encoded_features.numpy()[0][0])
                encoded_features_Y.append(encoded_features.numpy()[0][1])
                y_l.append(y.numpy()[0])

                if visualise[y.numpy()[0]]:
                    self.vis.visualisingReconstructedSpike(X.numpy().flatten(),
                                                      reconstructed_spike.numpy().flatten(),
                                                      len(X.numpy().flatten()),
                                                      str(y.numpy()[0]))
                    self.vis.printSpike(X.numpy().flatten(), len(X.numpy().flatten()), "b", f"real_spike{y.numpy()[0]}")
                    self.vis.printSpike(reconstructed_spike.numpy().flatten(),
                                   len(reconstructed_spike.numpy().flatten()), "r",
                                   f"reconstructed_spike{y.numpy()[0]}")
                    visualise[y.numpy()[0]] = False

        self.logger.info(f"Number of Samples after Autoencoder testing: {len(encoded_features_list)}")
        self.logger.info(f"First Spike after testing: {encoded_features_list[0]}")

        kmeans = KMeans(n_clusters=number_of_clusters)
        kmeans.fit(encoded_features_list)

        self.logger.info(y_test)
        self.logger.info(kmeans.labels_)
        for n in range(0, number_of_clusters):
            self.logger.info(f"Cluster {n} Occurrences: {(y_test == n).sum()}; KMEANS: {(kmeans.labels_ == n).sum()}")

        self.vis.visualisingClusters(encoded_features_X, encoded_features_Y, kmeans.labels_, kmeans.cluster_centers_)

        self.vis.printConfusionMatrix(y_l, kmeans.labels_, np.unique(y_test))
