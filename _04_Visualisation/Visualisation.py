import os
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import sort
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.metrics.cluster import contingency_matrix


class Visualisation(object):
    def __init__(self, variant_name, dataset_name):
        self.variant_name = variant_name
        self.dataset_name = dataset_name
        self.logger = None
        self.timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
        if os.path.exists(f"{self.dataset_name[0]}_{self.dataset_name[1]}") is False:
            os.mkdir(f"{self.dataset_name[0]}_{self.dataset_name[1]}")
        if os.path.exists(f"{self.dataset_name[0]}_{self.dataset_name[1]}/{self.variant_name}") is False:
            os.mkdir(f"{self.dataset_name[0]}_{self.dataset_name[1]}/{self.variant_name}")
        self.path = f"{self.dataset_name[0]}_{self.dataset_name[1]}/{self.variant_name}/{self.timestamp}"
        os.mkdir(self.path)

    def getVisualisationPath(self):
        return self.path

    def setLogger(self, logger):
        self.logger = logger

    def printSpike(self, spike, n_features, color, filename):
        plt.figure(figsize=(4, 4))
        t = np.arange(0, n_features, 1)
        plt.plot(t, spike, color=color)
        plt.savefig(f"{self.path}/{filename}.png")

    def visualisingReconstructedSpike(self, original, reconstructed, n_features, cluster):
        plt.figure(figsize=(4, 4))
        t = np.arange(0, n_features, 1)
        title = "True Label: " + cluster
        plt.plot(t, original, label="original")
        plt.plot(t, reconstructed, label="reconstructed")
        plt.legend(loc="upper left")
        plt.title(title)
        plt.savefig(f"{self.path}/reconstructedSpike_cluster_{cluster}.png")

    def getClusterCenters(self, features_list, labels):
        centroids = []
        unique_cluster = sort(np.unique(labels))
        for cluster in unique_cluster:
            x_list = []
            y_list = []
            for index in range(0, len(labels)):
                if cluster == labels[index]:
                    x_list.append(features_list[index][0])
                    y_list.append(features_list[index][1])
            x_mean = np.mean(x_list)
            y_mean = np.mean(y_list)
            centroids.append([x_mean, y_mean])
        self.logger.info(f"Centroids: {centroids}")
        return np.asarray(centroids)

    def visualisingClusters(self, x, y, cluster, centroids=None):
        plt.figure(figsize=(8, 8))
        unique_cluster = np.unique(cluster)
        for i in unique_cluster:
            x_i = []
            y_i = []
            for j in range(0, len(cluster)):
                if i == cluster[j]:
                    x_i.append(x[j])
                    y_i.append(y[j])
            plt.scatter(x_i, y_i, label=i)
        if centroids is None:
            pass
        else:
            plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k', marker='+')
        plt.legend(loc="upper left")
        plt.savefig(f"{self.path}/clusters.png")

    def printLossCurve(self, loss_values, epoch=""):
        plt.figure(figsize=(6, 4))
        t = np.arange(0, len(loss_values), 1)
        plt.plot(t, loss_values, color="b")
        plt.title(f"LossCurve{epoch}")
        if epoch == "":
            plt.xlabel("Epochs")
            plt.ylabel("Losses")
            plt.savefig(f"{self.path}/lossCurveEpochs.png")
        else:
            plt.xlabel("Each 100 Spikes")
            plt.ylabel("Losses")
            plt.savefig(f"{self.path}/lossCurveEpoch{epoch}.png")

    def printConfusionMatrix(self, ground_truth, predictions, labels):
        cm = contingency_matrix(ground_truth, predictions)
        self.logger.info("Contingency Matrix: ")
        self.logger.info(cm)
        print("Contingency Matrix: ")
        print(cm)

        plt.figure(figsize=(12, 6))
        plt.imshow(cm, interpolation="nearest")
        for (j, i), label in np.ndenumerate(cm):
            if label < 50:
                color = "yellow"
            else:
                color = "black"
            plt.text(i, j, label, ha="center", va="center", fontsize=13, color=color)
        plt.title("Contingency Matrix", fontsize=16)
        plt.xticks(sorted(np.unique(predictions)), fontsize=13)
        plt.yticks(sorted(labels), fontsize=13)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=13)
        plt.xlabel("Predicted label", fontsize=13)
        plt.ylabel("True label", fontsize=13)
        plt.savefig(f"{self.path}/contingency_matrix.png")

        mapping = []
        for predicted in range(0, len(np.unique(predictions))):
            get_truth_values = []
            for true in labels:
                get_truth_values.append(cm[true][predicted])
            mapping.append(np.argmax(get_truth_values))

        self.logger.info(f"Mapping: {mapping}")
        print(f"Mapping: {mapping}")

        predictions_mapping = []
        for prediction in predictions:
            predictions_mapping.append(mapping[prediction])

        cm_mapping = confusion_matrix(ground_truth, predictions_mapping, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_mapping, display_labels=labels)
        disp.plot()
        plt.title("With Cluster Mapping")
        plt.savefig(f"{self.path}/confusion_matrix.png")

        target_names = []
        for label in labels:
            target_names.append(f"cluster_{label}")

        self.logger.info(f"Accuracy: {accuracy_score(ground_truth, predictions_mapping)}")
        print(f"Accuracy: {accuracy_score(ground_truth, predictions_mapping)}")
        cr = classification_report(ground_truth, predictions_mapping, target_names=target_names)
        self.logger.info(cr)
        print(cr)
