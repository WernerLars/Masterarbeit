import matplotlib.pyplot as plt
import numpy as np

from LoadingDataset import LoadDataset
from autoencoder import Autoencoder
from SpikeClassToPytorchDataset import SpikeClassToPytorchDataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from sklearn.cluster import KMeans

path = "_00_Datasets/03_SimDaten_Quiroga2020/004_C_Difficult1_noise005.mat"

dataset = LoadDataset()
dataloader, y_labels = dataset.loadData(path)
input_size = len(dataloader.aligned_spikes[0])
print("Input Size: ", input_size)

train_data = SpikeClassToPytorchDataset(dataloader.aligned_spikes, y_labels)
train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
print(train_dataloader)

model = Autoencoder(input_size)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

encoded_features_list = []


def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        # Compute prediction error
        reconstructed_spike, encoded_features = model(X)
        with torch.no_grad():
            if epoch == 4:
                encoded_features_list.append(encoded_features.numpy()[0])
        loss = loss_fn(reconstructed_spike, X)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, t)
print("Done!")

print("Number of Samples after Autoencoder training: ", len(encoded_features_list))
print("First Spike after Autoencoder: ", encoded_features_list[0])

number_of_clusters = max(y_labels)+1

print("Number of Clusters: ", number_of_clusters)

kmeans = KMeans(n_clusters=number_of_clusters)
kmeans.fit(encoded_features_list)

print(kmeans.labels_)
for i in range(0, number_of_clusters):
    print("Cluster ", i, " Occurences: ", (y_labels == i).sum(), "; KMEANS: ", (kmeans.labels_ == i).sum())
