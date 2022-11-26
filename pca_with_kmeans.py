from _00_Extern_Functions.load_mat_files import load_mat_file
from _00_Extern_Functions.spike_class import spike_dataclass
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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


pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(dataloader.aligned_spikes)
print("Number of Samples after PCA: ", len(pca_transformed))
print("First Spike Frame after PCA: ", pca_transformed[0])

kmeans = KMeans(n_clusters=max(dataloader.cluster))
kmeans.fit(pca_transformed)

print(kmeans.labels_)


