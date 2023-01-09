import numpy as np

from _01_LoadDataset.LoadingDataset import LoadDataset
from _01_LoadDataset.SpikeClassToPytorchDataset import SpikeClassToPytorchDataset
from _02_Classes_Autoencoder_QLearning.Autoencoder import *
from _02_Classes_Autoencoder_QLearning.QLearning import Q_Learning
from torch.utils.data import DataLoader
import torch
from torch import nn


def Variant_04_Offline_Autoencoder_QLearning(path, vis, logger):
    dataset = LoadDataset()
    data, y_labels = dataset.loadData(path, logger)
    input_size = len(data.aligned_spikes[0])
    logger.info(f"Input Size: {input_size}")

    train_idx = round(len(data.aligned_spikes) * 0.8)
    logger.info(f"Train Index: {train_idx}")

    x_train = data.aligned_spikes[0:train_idx]
    y_train = y_labels[0:train_idx]
    x_test = data.aligned_spikes[train_idx:]
    y_test = y_labels[train_idx:]

    logger.info(f"x_train: {len(x_train)}")
    logger.info(f"y_train: {len(y_train)}")
    logger.info(f"x_test: {len(x_test)}")
    logger.info(f"y_test: {len(y_test)}")

    train_d = SpikeClassToPytorchDataset(x_train, y_train)
    train_dl = DataLoader(train_d, batch_size=1, shuffle=True)
    logger.info(train_dl)
    test_d = SpikeClassToPytorchDataset(x_test, y_test)
    test_dl = DataLoader(test_d, batch_size=1, shuffle=True)
    logger.info(test_dl)

    autoencoder = Autoencoder(input_size)
    # autoencoder = ConvolutionalAutoencoder(input_size)
    logger.info(autoencoder)

    ql = Q_Learning()

    loss_function = nn.MSELoss()
    adam = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

    epochs = 8
    for t in range(epochs):
        logger.info(f"Epoch {t + 1}\n-------------------------------")
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dl, autoencoder, loss_function, adam, logger)

    test(test_dl, y_test, autoencoder, ql, vis, logger)
    logger.info("Done!")


def train(dataloader, model, loss_fn, optimizer, logger):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        # Compute prediction error
        reconstructed_spike, encoded_features = model(X)
        loss = loss_fn(reconstructed_spike, X)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss Computation
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, y_test, model, ql, vis, logger):
    encoded_features_list = []
    encoded_features_X = []
    encoded_features_Y = []
    y_l = []
    ql.reset_spikes_clusters()

    number_of_clusters = max(y_test) + 1
    logger.info(f"Number of Clusters: {number_of_clusters}")

    visualise = []
    firstTwoSpikes = 0
    for k in range(0, number_of_clusters):
        visualise.append(True)

    current = 1
    size = len(y_test)-2
    for batch, (X, y) in enumerate(dataloader):
        reconstructed_spike, encoded_features = model(X)

        # First Two Spikes are just added to FeatureSet to make normalisation work
        with torch.no_grad():
            if firstTwoSpikes < 2:
                ql.addToFeatureSet(encoded_features.numpy()[0])
                firstTwoSpikes += 1
            else:

                ql.dynaQAlgorithm(encoded_features.numpy()[0])
                logger.info(f"Q_Learning: {current:>5d}/{size:>5d}]")
                print(f"Q_Learning: {current:>5d}/{size:>5d}]")
                current += 1
                encoded_features_list.append(encoded_features.numpy()[0])
                encoded_features_X.append(encoded_features.numpy()[0][0])
                encoded_features_Y.append(encoded_features.numpy()[0][1])
                y_l.append(y.numpy()[0])

                # Visualisation of Real Spike to Reconstructed Spike on Ground Truth Data
                if visualise[y.numpy()[0]]:
                    vis.visualisingReconstructedSpike(X.numpy().flatten(),
                                                      reconstructed_spike.numpy().flatten(),
                                                      len(X.numpy().flatten()),
                                                      str(y.numpy()[0]))
                    visualise[y.numpy()[0]] = False

    logger.info(f"Number of Samples after Autoencoder testing: {len(encoded_features_list)}")
    logger.info(f"First Spike after testing: {encoded_features_list[0]}")

    logger.info(y_l)
    logger.info(ql.clusters)

    # Visualisation only with the last 100 Spikes
    centroids = vis.getClusterCenters(encoded_features_list[-100:], ql.clusters[-100:])
    vis.visualisingClusters(encoded_features_X[-100:], encoded_features_Y[-100:], ql.clusters[-100:], centroids)

    vis.printConfusionMatrix(y_l, ql.clusters, np.unique(y_l), logger)
