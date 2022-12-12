import numpy as np

from _01_LoadDataset.LoadingDataset import LoadDataset
from sklearn.decomposition import PCA
from _02_Classes_Autoencoder_QLearning.QLearning import Q_Learning
from _04_Visualisation.Visualisation import visualisingClusters


def Variant_03_PCA_QLearning(path):
    dataset = LoadDataset()
    dataloader, y_labels = dataset.loadData(path)

    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(dataloader.aligned_spikes)
    print("Number of Samples after PCA: ", len(pca_transformed))
    print("First Spike Frame after PCA: ", pca_transformed[0])

    ql = Q_Learning()
    for s in range(0, 2):
        ql.addToFeatureSet(pca_transformed[s])

    for s in range(2, 402):
        ql.dynaQAlgorithm(pca_transformed[s])
        print(ql.q_table)
        print(ql.model)
    x = []
    y = []
    for spike in ql.spikes:
        x.append(spike[0])
        y.append(spike[1])

    print(ql.clusters)
    print(ql.randomFeatures)
    visualisingClusters(x, y, ql.clusters)
