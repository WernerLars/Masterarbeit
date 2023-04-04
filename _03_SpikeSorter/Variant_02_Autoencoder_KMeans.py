from tqdm import tqdm

from _01_LoadDataset.LoadingDataset import LoadDataset
from _01_LoadDataset.SpikeClassToPytorchDataset import SpikeClassToPytorchDataset
from _02_Classes_Autoencoder_QLearning.Autoencoder import *
from torch.utils.data import DataLoader
import torch
from torch import nn
from sklearn.cluster import KMeans


class Variant_02_Autoencoder_KMeans(object):
    def __init__(self, path, vis, logger, parameter_logger,
                 chooseAutoencoder=1, epochs=8, batch_size=1, number_of_features=2):
        self.path = path
        self.vis = vis
        self.logger = logger
        self.parameter_logger = parameter_logger
        self.chooseAutoencoder = chooseAutoencoder
        self.epochs = epochs
        self.parameter_logger.info(f"Epochs: {self.epochs}")
        self.batch_size = batch_size
        self.parameter_logger.info(f"Batch Size: {self.batch_size}")

        self.dataset = LoadDataset(self.path, self.logger)
        self.data, self.y_labels = self.dataset.load_data()
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
            self.train(dataloader, t+1)
            self.logger.info(f"Epoch [{t + 1}/{self.epochs}]: mean_loss={self.loss_values[t]}")
            self.vis.print_loss_curve(self.epoch_loss, t+1)

        self.vis.print_loss_curve(self.loss_values)
        self.clustering(dataloader, self.y_labels)
        self.logger.info("Done!")

    def train(self, train_dataloader, epoch_number):
        self.autoencoder.train()
        self.epoch_loss = []
        training_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
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
        self.loss_values.append(sum(self.epoch_loss) / len(train_dataloader))

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
                    self.vis.visualising_reconstructed_spike(x_np,
                                                             reconstructed_spike_np,
                                                             len(x_np),
                                                             str(cluster))
                    self.vis.print_spike(x_np, len(x_np),
                                         "b", f"real_spike{cluster}")
                    self.vis.print_spike(reconstructed_spike_np, len(reconstructed_spike_np),
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

        centroids_true = self.vis.get_cluster_centers(encoded_features_list, cluster_labels)
        centroids_kmeans = self.vis.get_cluster_centers(encoded_features_list, kmeans.labels_)

        self.vis.visualising_features(encoded_features_X, encoded_features_Y)
        self.vis.visualising_clusters(encoded_features_X, encoded_features_Y,
                                      cluster_labels, centroids_true, "true")
        self.vis.visualising_clusters(encoded_features_X, encoded_features_Y,
                                      kmeans.labels_, centroids_kmeans, "kmeans")

        self.vis.print_metrics(cluster_labels, kmeans.labels_)
