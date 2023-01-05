from _01_LoadDataset.LoadingDataset import LoadDataset
from _01_LoadDataset.SpikeClassToPytorchDataset import SpikeClassToPytorchDataset
from _02_Classes_Autoencoder_QLearning.Autoencoder import *
from _02_Classes_Autoencoder_QLearning.QLearning import Q_Learning
from _04_Visualisation.Visualisation import *
from torch.utils.data import DataLoader
import torch
from torch import nn


def Variant_04_Autoencoder_QLearning(path):
    dataset = LoadDataset()
    data, y_labels = dataset.loadData(path)
    input_size = len(data.aligned_spikes[0])
    print(f"Input Size: {input_size}")

    train_idx = round(len(data.aligned_spikes) * 0.8)
    print(f"Train Index: {train_idx}")

    x_train = data.aligned_spikes[0:train_idx]
    y_train = y_labels[0:train_idx]
    x_test = data.aligned_spikes[train_idx:]
    y_test = y_labels[train_idx:]

    print(f"x_train: {len(x_train)}")
    print(f"y_train: {len(y_train)}")
    print(f"x_test: {len(x_test)}")
    print(f"y_test: {len(y_test)}")

    train_d = SpikeClassToPytorchDataset(x_train, y_train)
    train_dl = DataLoader(train_d, batch_size=1, shuffle=True)
    print(train_dl)
    test_d = SpikeClassToPytorchDataset(x_test, y_test)
    test_dl = DataLoader(test_d, batch_size=1, shuffle=True)
    print(test_dl)

    autoencoder = Autoencoder(input_size)
    #autoencoder = ConvolutionalAutoencoder(input_size)
    print(autoencoder)

    ql = Q_Learning()

    loss_function = nn.MSELoss()
    adam = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

    epochs = 8
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dl, autoencoder, loss_function, adam)

    test(test_dl, y_test, autoencoder, ql)
    print("Done!")


def train(dataloader, model, loss_fn, optimizer):
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
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, y_test, model, ql):
    encoded_features_list = []
    encoded_features_X = []
    encoded_features_Y = []
    y_l = []
    ql.reset_spikes_clusters()

    number_of_clusters = max(y_test) + 1
    print(f"Number of Clusters: {number_of_clusters}")

    visualise = []
    firstTwoSpikes = 0
    for k in range(0, number_of_clusters):
        visualise.append(True)

    for batch, (X, y) in enumerate(dataloader):
        reconstructed_spike, encoded_features = model(X)

        # First Two Spikes are just added to FeatureSet to make normalisation work
        with torch.no_grad():
            if firstTwoSpikes < 2:
                ql.addToFeatureSet(encoded_features.numpy()[0])
                firstTwoSpikes += 1
            else:

                ql.dynaQAlgorithm(encoded_features.numpy()[0])
                print(ql.q_table)
                print(ql.model)
                encoded_features_list.append(encoded_features.numpy()[0])
                encoded_features_X.append(encoded_features.numpy()[0][0])
                encoded_features_Y.append(encoded_features.numpy()[0][1])
                y_l.append(y.numpy()[0])

                # Visualisation of Real Spike to Reconstructed Spike on Ground Truth Data
                if visualise[y.numpy()[0]]:
                    visualisingReconstructedSpike(X.numpy().flatten(),
                                                  reconstructed_spike.numpy().flatten(),
                                                  len(X.numpy().flatten()),
                                                  str(y.numpy()[0]))
                    visualise[y.numpy()[0]] = False

    print(f"Number of Samples after Autoencoder testing: {len(encoded_features_list)}")
    print(f"First Spike after testing: {encoded_features_list[0]}")

    print(y_l)
    print(ql.clusters)

    # Visualisation only with the last 100 Spikes
    centroids = getClusterCenters(encoded_features_list[-100:], ql.clusters[-100:])
    visualisingClusters(encoded_features_X[-100:], encoded_features_Y[-100:], ql.clusters[-100:], centroids)

    printConfusionMatrix(y_l, ql.clusters, np.unique(y_l))
