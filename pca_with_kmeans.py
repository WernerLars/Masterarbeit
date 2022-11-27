from LoadingDataset import LoadDataset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

path = "_00_Datasets/03_SimDaten_Quiroga2020/004_C_Difficult1_noise005.mat"

dataset = LoadDataset()
dataloader, y_labels = dataset.loadData(path)

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(dataloader.aligned_spikes)
print("Number of Samples after PCA: ", len(pca_transformed))
print("First Spike Frame after PCA: ", pca_transformed[0])

number_of_clusters = max(y_labels)+1
print("Number of Clusters: ", number_of_clusters)

kmeans = KMeans(n_clusters=number_of_clusters)
kmeans.fit(pca_transformed)

print(kmeans.labels_)
for i in range(0, number_of_clusters):
    print("Cluster ", i, " Occurences: ", (y_labels == i).sum(), "; KMEANS: ", (kmeans.labels_ == i).sum())


