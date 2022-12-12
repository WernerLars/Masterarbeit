import numpy as np

from _01_LoadDataset.LoadingDataset import LoadDataset
from _04_Visualisation.Visualisation import visualisingClusters, printConfusionMatrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def Variant_01_PCA_KMeans(path):
    dataset = LoadDataset()
    dataloader, y_labels = dataset.loadData(path)

    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(dataloader.aligned_spikes)
    print("Number of Samples after PCA: ", len(pca_transformed))
    print("First Spike Frame after PCA: ", pca_transformed[0])

    number_of_clusters = max(y_labels) + 1
    print("Number of Clusters: ", number_of_clusters)

    kmeans = KMeans(n_clusters=number_of_clusters)
    kmeans.fit(pca_transformed)

    print(kmeans.labels_)
    for i in range(0, number_of_clusters):
        print("Cluster ", i, " Occurences: ", (y_labels == i).sum(), "; KMEANS: ", (kmeans.labels_ == i).sum())

    x = []
    y = []
    for spike in pca_transformed:
        x.append(spike[0])
        y.append(spike[1])

    print(kmeans.cluster_centers_)

    visualisingClusters(x, y, kmeans.labels_, kmeans.cluster_centers_)

    printConfusionMatrix(y_labels, kmeans.labels_, np.unique(y_labels))
