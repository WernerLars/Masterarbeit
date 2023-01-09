import numpy as np

from _01_LoadDataset.LoadingDataset import LoadDataset
from sklearn.decomposition import PCA
from _02_Classes_Autoencoder_QLearning.QLearning import Q_Learning


def Variant_03_PCA_QLearning(path, vis, logger):
    dataset = LoadDataset()
    dataloader, y_labels = dataset.loadData(path, logger)

    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(dataloader.aligned_spikes)
    logger.info(f"Number of Samples after PCA: {len(pca_transformed)}")
    logger.info(f"First Spike Frame after PCA: {pca_transformed[0]}")

    ql = Q_Learning()
    for s in range(0, 2):
        ql.addToFeatureSet(pca_transformed[s])

    size = 300
    logger.info(f"Number of Spikes for Q_Learning: {size}")
    for s in range(0, size):
        ql.dynaQAlgorithm(pca_transformed[s])
        logger.info(f"Q_Learning: {s:>5d}/{size:>5d}]")
        print(f"Q_Learning: {s:>5d}/{size:>5d}]")
    x = []
    y = []
    for spike in ql.spikes:
        x.append(spike[0])
        y.append(spike[1])

    logger.info(ql.clusters)
    logger.info(ql.randomFeatures)

    centroids = vis.getClusterCenters(ql.spikes, ql.clusters)
    vis.visualisingClusters(x, y, ql.clusters, centroids)
    vis.printConfusionMatrix(y_labels[:size], ql.clusters, np.unique(y_labels[:size]), logger)
