from tqdm import tqdm

from _01_LoadDataset.LoadingDataset import LoadDataset
from _01_LoadDataset.SpikeClassToPytorchDataset import SpikeClassToPytorchDataset
from _02_Classes_Autoencoder_QLearning.Autoencoder import *
from _02_Classes_Autoencoder_QLearning.QLearning import Q_Learning
from torch.utils.data import DataLoader
import torch
from torch import nn


class Variant_04_Offline_Autoencoder_QLearning(object):
    def __init__(self, path, vis, logger, parameter_logger, normalise=False,
                 chooseAutoencoder=1, split_ratio=0.9, epochs=8, batch_size=1,
                 number_of_features=2,
                 punishment_coefficient=0.6):
        self.path = path
        self.vis = vis
        self.logger = logger
        self.parameter_logger = parameter_logger
        self.normalise = normalise
        self.chooseAutoencoder = chooseAutoencoder
        self.split_ratio = split_ratio
        self.parameter_logger.info(f"Split Ratio: {self.split_ratio}")
        self.epochs = epochs
        self.parameter_logger.info(f"Epochs: {self.epochs}")
        self.batch_size = batch_size
        self.parameter_logger.info(f"Batch Size: {self.batch_size}")
        self.logger.info(f"Punishment_Coefficient: {punishment_coefficient}")

        self.dataset = LoadDataset(self.path, self.logger)
        self.data, self.y_labels = self.dataset.load_data()
        self.input_size = len(self.data.aligned_spikes[0])
        self.parameter_logger.info(f"Input Size: {self.input_size}")
        self.number_of_features = number_of_features
        self.parameter_logger.info(f"Number of Features: {self.number_of_features}")

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

        self.ql = Q_Learning(self.parameter_logger, self.number_of_features,
                             normalise=self.normalise,
                             punishment_coefficient=punishment_coefficient)

        self.loss_values = []
        self.epoch_loss = []

        self.preprocessing()

    def preprocessing(self):
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
            self.train(train_dl, t+1)
            self.logger.info(f"Epoch [{t + 1}/{self.epochs}]: mean_loss={self.loss_values[t]}")
            self.vis.print_loss_curve(self.epoch_loss, t+1)

        self.vis.print_loss_curve(self.loss_values)
        self.test(test_dl, y_test)
        self.logger.info("Done!")

    def train(self, dataloader, epoch_number):
        self.autoencoder.train()
        self.epoch_loss = []
        training_loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch, (X, y) in training_loop:

            # Compute reconstruction error
            reconstructed_spike, encoded_features = self.autoencoder(X)
            loss = self.loss_function(reconstructed_spike, X)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Loss Computation
            if batch % 100 == 0:
                loss = loss.item()
                training_loop.set_description(f"Epoch [{epoch_number}/{self.epochs}]")
                training_loop.set_postfix(loss=loss)
                self.epoch_loss.append(loss)
        self.loss_values.append(sum(self.epoch_loss) / len(dataloader))

    def test(self, dataloader, y_test):
        self.ql.reset_spikes_clusters()

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

        q_learning_loop = tqdm(enumerate(dataloader), total=len(dataloader))
        q_learning_loop.set_description(f"Q_Learning")
        for batch, (X, y) in q_learning_loop:
            reconstructed_spike, encoded_features = self.autoencoder(X)

            with torch.no_grad():
                # If Normalisation then first two Spikes are added to FeatureSet to make normalisation work
                if firstTwoSpikes < 2 and self.normalise:
                    self.ql.add_to_feature_set(encoded_features.numpy()[0])
                    firstTwoSpikes += 1
                else:
                    self.ql.dyna_q_algorithm(encoded_features.numpy()[0])

                    encoded_features_list.append(encoded_features.numpy()[0])
                    encoded_features_X.append(encoded_features.numpy()[0][0])
                    encoded_features_Y.append(encoded_features.numpy()[0][1])
                    cluster = y.numpy()[0]
                    cluster_labels.append(cluster)

                    # Visualisation of Real Spike to Reconstructed Spike on Ground Truth Data
                    if visualise[cluster]:
                        self.vis.visualising_reconstructed_spike(X.numpy().flatten(),
                                                                 reconstructed_spike.numpy().flatten(),
                                                                 len(X.numpy().flatten()),
                                                                 str(cluster))
                        visualise[cluster] = False

        self.logger.info(f"Number of Samples after Autoencoder testing: {len(encoded_features_list)}")
        self.logger.info(f"First Spike after testing: {encoded_features_list[0]}")

        self.logger.info(cluster_labels)
        self.logger.info(self.ql.clusters)
        self.ql.print_q_table()
        self.ql.print_model()

        centroids_true = self.vis.get_cluster_centers(encoded_features_list, cluster_labels)
        centroids_qlearning = self.vis.get_cluster_centers(encoded_features_list, self.ql.clusters)

        self.vis.visualising_features(encoded_features_X, encoded_features_Y)

        self.vis.visualising_clusters(encoded_features_X, encoded_features_Y, cluster_labels,
                                      centroids_true, "true")
        self.vis.visualising_clusters(encoded_features_X, encoded_features_Y, self.ql.clusters,
                                      centroids_qlearning, "qlearning")
        self.vis.print_metrics(cluster_labels, self.ql.clusters)

        if self.normalise:
            centroids_qlearning_normalised = self.vis.get_cluster_centers(self.ql.spikes, self.ql.clusters)
            x_norm = []
            y_norm = []
            for features_normalised in self.ql.spikes:
                x_norm.append(features_normalised[0])
                y_norm.append(features_normalised[1])
            self.vis.visualising_features(x_norm, y_norm, "_normalised")
            self.vis.visualising_clusters(x_norm, y_norm, cluster_labels,
                                          filename="true_normalised")
            self.vis.visualising_clusters(x_norm, y_norm, self.ql.clusters,
                                          centroids_qlearning_normalised, "qlearning_normalised")
