from numpy.random import uniform

from _01_LoadDataset.LoadingDataset import LoadDataset
from _01_LoadDataset.SpikeClassToPytorchDataset import SpikeClassToPytorchDataset
from _02_Classes_Autoencoder_QLearning.Autoencoder import *
from _02_Classes_Autoencoder_QLearning.QLearning import Q_Learning
from _02_Classes_Autoencoder_QLearning.Templates import Templates
from torch.utils.data import DataLoader
import torch
from torch import nn


class Variant_05_Online_Autoencoder_QLearning(object):
    def __init__(self, path, vis, logger, parameter_logger, normalise=False, templateMatching=False, optimising=False,
                 noisyBatch=False, updateFactor=100, noiseFactor=0.1,
                 chooseAutoencoder=1, epochs=8, batch_size=1,
                 maxAutoencoderTraining=700, maxTraining=1000,
                 number_of_features=2,
                 punishment_coefficient=0.6):
        self.path = path
        self.vis = vis
        self.logger = logger
        self.parameter_logger = parameter_logger
        self.chooseAutoencoder = chooseAutoencoder
        self.epochs = epochs
        self.normalise = normalise
        self.parameter_logger.info(f"Normalisation: {self.normalise}")
        self.templateMatching = templateMatching
        self.parameter_logger.info(f"Template Matching: {self.templateMatching}")
        self.optimising = optimising
        self.parameter_logger.info(f"Optimising Autoencoder: {self.optimising}")
        self.updateFactor = updateFactor
        self.parameter_logger.info(f"Update Factor: {self.updateFactor}")
        self.noisyBatch = noisyBatch
        self.parameter_logger.info(f"Noisy Batches: {self.noisyBatch}")
        self.noiseFactor = noiseFactor
        self.parameter_logger.info(f"Noisy Factor: {self.noiseFactor}")
        self.parameter_logger.info(f"Epochs: {self.epochs}")
        self.batch_size = batch_size
        self.parameter_logger.info(f"Batch Size: {self.batch_size}")
        self.maxAutoencoderTraining = maxAutoencoderTraining
        self.parameter_logger.info(f"maximal Spikes for Autoencoder Training : {self.maxAutoencoderTraining}")
        self.maxTraining = maxTraining
        self.parameter_logger.info(f"maximal Spikes for Training: {self.maxTraining}")
        self.logger.info(f"Punishment_Coefficient: {punishment_coefficient}")

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
        self.optimisingAutoencoder = self.autoencoder_models[self.chooseAutoencoder][1]

        self.loss_function = nn.MSELoss()
        self.parameter_logger.info(self.loss_function)

        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        self.parameter_logger.info(self.optimizer)

        self.templates = Templates()
        self.ql = Q_Learning(self.parameter_logger, self.number_of_features,
                             normalise=self.normalise,
                             punishment_coefficient=punishment_coefficient)

        self.encoded_features_list = []
        self.encoded_features_X = []
        self.encoded_features_Y = []
        self.cluster_labels = []
        self.firstTwoSpikes = 0
        self.visualise = []
        self.number_of_clusters = max(self.y_labels) + 1
        self.logger.info(f"Number of Clusters: {self.number_of_clusters}")
        for k in range(0, self.number_of_clusters):
            self.visualise.append(True)

        self.loss_values = []
        self.epoch_loss = []

        self.preprocessing()
        self.clusterVisualisation()

    def preprocessing(self):

        data = SpikeClassToPytorchDataset(self.data.aligned_spikes, self.y_labels)
        dataloader = DataLoader(data, batch_size=self.batch_size)

        for t, (X, y) in enumerate(dataloader):
            self.logger.info(f"Spike: {t}\n-------------------------------")
            print(f"Spike: {t}\n-------------------------------")
            self.epoch_loss = []
            if t < self.maxAutoencoderTraining:
                for _ in range(self.epochs):
                    self.train(X, self.autoencoder)
                t += 1
            elif t < self.maxTraining:
                if t % self.updateFactor == 0 and self.optimising:
                    self.autoencoder.load_state_dict(torch.load(f"{self.vis.path}/model.pt"))
                    print(f"{t}: Model updated")
                self.trainAutoencoderWithQLearning(X, y)
                t += 1
            else:
                break
            self.loss_values.append(sum(self.epoch_loss) / self.epochs)
        self.vis.printLossCurve(self.loss_values)

    def train(self, batch, model):
        model.train()
        for X in batch:
            # Compute reconstruction error
            reconstructed_spike, encoded_features = model(X)
            loss = self.loss_function(reconstructed_spike, X)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Loss Computation
            loss = loss.item()
            self.epoch_loss.append(loss)
            self.logger.info(f"loss: {loss:>7f}")
            print(f"loss: {loss:>7f}")
        if self.optimising:
            torch.save(model.state_dict(), f"{self.vis.path}/model.pt")

    def trainAutoencoderWithQLearning(self, X, y):
        reconstructed_spike, encoded_features = self.autoencoder(X)
        batch = X
        if self.firstTwoSpikes < 2 and self.normalise:
            with torch.no_grad():
                self.ql.addToFeatureSet(encoded_features.numpy()[0])
                self.firstTwoSpikes += 1
        else:
            with torch.no_grad():
                cluster = self.ql.dynaQAlgorithm(encoded_features.numpy()[0])

                self.encoded_features_list.append(encoded_features.numpy()[0])
                self.encoded_features_X.append(encoded_features.numpy()[0][0])
                self.encoded_features_Y.append(encoded_features.numpy()[0][1])
                true_label = y.numpy()[0]
                self.cluster_labels.append(true_label)

                if self.visualise[true_label]:
                    self.vis.visualisingReconstructedSpike(X.numpy().flatten(),
                                                           reconstructed_spike.numpy().flatten(),
                                                           len(X.numpy().flatten()),
                                                           str(true_label))
                    self.visualise[true_label] = False

        if self.templateMatching:
            self.templates.computeMeanTemplate(X, cluster)
            batch = []
            for c_index in range(0, len(self.templates.template_list)):
                if c_index == cluster:
                    batch.append(X)
                else:
                    template = self.templates.template_list[c_index]
                    if self.noisyBatch:
                        for dim, value in enumerate(template):
                            p = uniform(-self.noiseFactor, self.noiseFactor)
                            template[dim] = value + p
                    batch.append(template)

        if self.optimising:
            self.optimisingAutoencoder.load_state_dict(torch.load(f"{self.vis.path}/model.pt"))
            self.train(batch, self.optimisingAutoencoder)

    def clusterVisualisation(self):

        self.logger.info(f"Number of Samples after Autoencoder testing: {len(self.encoded_features_list)}")
        self.logger.info(f"First Spike after testing: {self.encoded_features_list[0]}")

        self.logger.info(self.cluster_labels)
        self.logger.info(self.ql.clusters)
        self.ql.printQTable()
        self.ql.printModel()

        centroids_true = self.vis.getClusterCenters(self.encoded_features_list, self.cluster_labels)
        centroids_qlearning = self.vis.getClusterCenters(self.encoded_features_list, self.ql.clusters)

        self.vis.visualisingFeatures(self.encoded_features_X, self.encoded_features_Y)
        self.vis.visualisingClusters(self.encoded_features_X, self.encoded_features_Y, self.cluster_labels,
                                     centroids_true, "true")
        self.vis.visualisingClusters(self.encoded_features_X, self.encoded_features_Y, self.ql.clusters,
                                     centroids_qlearning, "qlearning")

        self.vis.printMetrics(self.cluster_labels, self.ql.clusters)
