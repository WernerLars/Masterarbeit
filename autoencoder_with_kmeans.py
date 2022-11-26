from _00_Extern_Functions.load_mat_files import load_mat_file
from _00_Extern_Functions.spike_class import spike_dataclass
from autoencoder import Autoencoder
from SpikeClassToPytorchDataset import SpikeClassToPytorchDataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
from torch import nn

loaded_data = load_mat_file("_00_Datasets/03_SimDaten_Quiroga2020/004_C_Difficult1_noise005.mat")

data = dict()
data['sampling_rate'] = float(1 / loaded_data['samplingInterval'][0][0] * 1000)
data['raw_data'] = loaded_data['data'][0]
data['spike_times'] = loaded_data['spike_times'][0][0][0]
data['spike_cluster'] = loaded_data['spike_class'][0][0][0]

dataloader = spike_dataclass(data)
dataloader.align_spike_frames()
print(dataloader)
print("Sampling rate: ", dataloader.sampling_rate)
print("Raw: ", dataloader.raw)
print("Times: ", dataloader.times)
print("Cluster: ", dataloader.cluster)
print("Number of different clusters: ", max(dataloader.cluster))
print("Number of Spikes: ", len(dataloader.cluster))
print("First aligned Spike Frame: ", dataloader.aligned_spikes[0])

y_labels = dataloader.cluster
y_labels[y_labels == 1] = 0
y_labels[y_labels == 2] = 1
y_labels[y_labels == 3] = 2

print("Cluster 0 Occurences: ", (y_labels == 0).sum())
print("Cluster 1 Occurences: ", (y_labels == 1).sum())
print("Cluster 2 Occurences: ", (y_labels == 2).sum())

input_size = len(dataloader.aligned_spikes[0])

print("Input Size: ", input_size)

model = Autoencoder(input_size)
print(model)

train_data = SpikeClassToPytorchDataset(dataloader.aligned_spikes, y_labels)

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False)

print(train_dataloader)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        # Compute prediction error
        reconstructed_image = model(X)
        loss = loss_fn(reconstructed_image, X)

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
    train(train_dataloader, model, loss_fn, optimizer)
print("Done!")

# kmeans = KMeans(n_clusters=max(dataloader.cluster))
# kmeans.fit(pca_transformed)

# print(kmeans.labels_)
