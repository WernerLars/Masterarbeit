import numpy as np

from _01_LoadDataset.LoadingDataset import LoadDataset
from _01_LoadDataset.SpikeClassToPytorchDataset import SpikeClassToPytorchDataset
from _02_Classes_Autoencoder_QLearning.Autoencoder import *
from _02_Classes_Autoencoder_QLearning.QLearning import Q_Learning
from _02_Classes_Autoencoder_QLearning.Templates import Templates
from torch.utils.data import DataLoader
import torch
from torch import nn


class Variant_05_Online_Autoencoder_QLearning(object):
    def __init__(self, path, vis, logger,
                 chooseAutoencoder=1, split_ratio=0.8, epochs=8, batch_size=1,
                 maxAutoencoderTraining=300, maxTraining=1000):
        self.path = path
        self.vis = vis
        self.logger = logger
        self.chooseAutoencoder = chooseAutoencoder
        self.split_ratio = split_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.maxAutoencoderTraining = maxAutoencoderTraining
        self.maxTraining = maxTraining

        self.dataset = LoadDataset(self.path, self.logger)
        self.data, self.y_labels = self.dataset.loadData()
        self.input_size = len(self.data.aligned_spikes[0])
        self.logger.info(f"Input Size: {self.input_size}")
        self.autoencoder_models = {
            1: Autoencoder(self.input_size),
            2: ConvolutionalAutoencoder(self.input_size)
        }
        self.autoencoder = self.autoencoder_models[self.chooseAutoencoder]
        self.logger.info(self.autoencoder)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        self.templates = Templates()
        self.ql = Q_Learning()

        self.preprocessing()

    def preprocessing(self):
        torch.manual_seed(0)
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

        t = 0
        firstTwoSpikes = 0
        for _, (X, _) in enumerate(train_dl):
            self.logger.info(f"Spike: {t}\n-------------------------------")
            print(f"Spike: {t}\n-------------------------------")
            if t < self.maxAutoencoderTraining:
                for _ in range(self.epochs):
                    self.train(X)
                t += 1
            elif firstTwoSpikes < 2:
                _, encoded_features = self.autoencoder(X)
                with torch.no_grad():
                    self.ql.addToFeatureSet(encoded_features.numpy()[0])
                firstTwoSpikes += 1
                t += 1
            elif t < self.maxTraining:
                self.trainAutoencoderWithQLearning(X)
                t += 1
            else:
                break

        self.test(test_dl, y_test)
        self.logger.info("Done!")

    def train(self, batch):
        self.autoencoder.train()
        for X in batch:
            # Compute prediction error
            reconstructed_spike, encoded_features = self.autoencoder(X)
            loss = self.loss_function(reconstructed_spike, X)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Loss Computation
            loss = loss.item()
            self.logger.info(f"loss: {loss:>7f}")
            print(f"loss: {loss:>7f}")

    def trainAutoencoderWithQLearning(self, spike):
        _, encoded_features = self.autoencoder(spike)
        with torch.no_grad():
            cluster = self.ql.dynaQAlgorithm(encoded_features.numpy()[0])
            self.templates.computeMeanTemplate(spike, cluster)
        batch = []
        for c_index in range(0, len(self.templates.template_list)):
            if c_index == cluster:
                batch.append(spike)
            else:
                batch.append(self.templates.template_list[c_index])
        self.train(batch)

    def test(self, dataloader, y_test):
        self.ql.reset_q_learning()

        encoded_features_list = []
        encoded_features_X = []
        encoded_features_Y = []
        cluster_labels = []

        number_of_clusters = max(y_test) + 1
        self.logger.info(f"Number of Clusters: {number_of_clusters}")

        visualise = []
        for k in range(0, number_of_clusters):
            visualise.append(True)

        firstTwoSpikes = 0
        current = 1
        size = len(y_test) - 2

        for batch, (X, y) in enumerate(dataloader):
            reconstructed_spike, encoded_features = self.autoencoder(X)

            # First Two Spikes are just added to FeatureSet to make normalisation work
            with torch.no_grad():
                if firstTwoSpikes < 2:
                    self.ql.addToFeatureSet(encoded_features.numpy()[0])
                    firstTwoSpikes += 1
                else:

                    self.ql.dynaQAlgorithm(encoded_features.numpy()[0])
                    self.logger.info(f"Q_Learning: {current:>5d}/{size:>5d}]")
                    print(f"Q_Learning: {current:>5d}/{size:>5d}]")
                    current += 1

                    encoded_features_list.append(encoded_features.numpy()[0])
                    encoded_features_X.append(encoded_features.numpy()[0][0])
                    encoded_features_Y.append(encoded_features.numpy()[0][1])
                    cluster = y.numpy()[0]
                    cluster_labels.append(cluster)

                    # Visualisation of Real Spike to Reconstructed Spike on Ground Truth Data
                    if visualise[cluster]:
                        self.vis.visualisingReconstructedSpike(X.numpy().flatten(),
                                                               reconstructed_spike.numpy().flatten(),
                                                               len(X.numpy().flatten()),
                                                               str(cluster))
                        visualise[cluster] = False

        self.logger.info(f"Number of Samples after Autoencoder testing: {len(encoded_features_list)}")
        self.logger.info(f"First Spike after testing: {encoded_features_list[0]}")

        self.logger.info(cluster_labels)
        self.logger.info(self.ql.clusters)

        centroids = self.vis.getClusterCenters(encoded_features_list, self.ql.clusters)
        self.vis.visualisingClusters(encoded_features_X, encoded_features_Y, self.ql.clusters, centroids)

        self.vis.printMetrics(cluster_labels, self.ql.clusters, np.unique(cluster_labels))
