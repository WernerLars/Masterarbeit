from LoadingDataset import LoadDataset
from autoencoder import Autoencoder
from SpikeClassToPytorchDataset import SpikeClassToPytorchDataset
from Visualisation import *
from torch.utils.data import DataLoader
import torch
from torch import nn
from sklearn.cluster import KMeans

# path = "_00_Datasets/01_SimDaten_Martinez2009/simulation_1.mat"
#path = "_00_Datasets/03_SimDaten_Quiroga2020/004_C_Difficult1_noise005.mat"
path = "_00_Datasets/03_SimDaten_Quiroga2020/016_C_Easy1_noise005.mat"

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
print(autoencoder)

loss_function = nn.MSELoss()
adam = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)


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

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    encoded_features_list = []
    encoded_features_X = []
    encoded_features_Y = []

    number_of_clusters = max(y_test) + 1
    print(f"Number of Clusters: {number_of_clusters}")

    visualise = []
    for k in range(0, number_of_clusters):
        visualise.append(True)

    for batch, (X, y) in enumerate(dataloader):
        reconstructed_spike, encoded_features = model(X)

        with torch.no_grad():
            encoded_features_list.append(encoded_features.numpy()[0])
            encoded_features_X.append(encoded_features.numpy()[0][0])
            encoded_features_Y.append(encoded_features.numpy()[0][1])

            if visualise[y.numpy()[0]]:
                visualisingReconstructedSpike(X.numpy().flatten(),
                                              reconstructed_spike.numpy().flatten(),
                                              len(X.numpy().flatten()),
                                              str(y.numpy()[0]))
                visualise[y.numpy()[0]] = False

        loss = loss_fn(reconstructed_spike, X)

    print(f"Number of Samples after Autoencoder testing: {len(encoded_features_list)}")
    print(f"First Spike after testing: {encoded_features_list[0]}")

    kmeans = KMeans(n_clusters=number_of_clusters)
    kmeans.fit(encoded_features_list)

    print(kmeans.labels_)
    for n in range(0, number_of_clusters):
        print(f"Cluster {n} Occurrences: {(y_test == n).sum()}; KMEANS: {(kmeans.labels_ == n).sum()}")

    visualisingClusters(encoded_features_X, encoded_features_Y, kmeans.labels_, kmeans.cluster_centers_)


epochs = 8
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dl, autoencoder, loss_function, adam)

test(test_dl, autoencoder, loss_function)
print("Done!")
