import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import sort
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.metrics.cluster import contingency_matrix


class Visualisation(object):
    def __init__(self, variant_name, dataset_name, exp_path=""):
        self.variant_name = variant_name
        self.dataset_name = dataset_name
        self.logger = None
        self.timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
        if os.path.exists(f"{exp_path}{self.dataset_name[1]}") is False:
            os.mkdir(f"{exp_path}{self.dataset_name[1]}")
        if os.path.exists(f"{exp_path}{self.dataset_name[1]}/{self.variant_name}") is False:
            os.mkdir(f"{exp_path}{self.dataset_name[1]}/{self.variant_name}")
        self.path = f"{exp_path}{self.dataset_name[1]}/{self.variant_name}/{self.timestamp}"
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

    def visualisingFeatures(self, x, y, filename=""):
        plt.figure(figsize=(8, 8))
        plt.scatter(x, y, color="k")
        plt.savefig(f"{self.path}/clusters_features{filename}.png")

    def visualisingClusters(self, x, y, cluster, centroids=None, filename=""):
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
        plt.savefig(f"{self.path}/clusters_{filename}.png")

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

    def printContingencyMatrix(self, cm, true_labels, predicted_labels, matched=None):
        plt.figure(figsize=(12, 6))
        plt.imshow(cm, interpolation="nearest")
        for (j, i), label in np.ndenumerate(cm):
            if label < 50:
                color = "yellow"
            else:
                color = "black"
            plt.text(i, j, label, ha="center", va="center", fontsize=13, color=color)
        plt.title("Contingency Matrix", fontsize=16)
        plt.xticks(ticks=np.arange(len(predicted_labels)), labels=predicted_labels, fontsize=13)
        plt.yticks(true_labels, fontsize=13)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=13)
        plt.xlabel("Predicted label", fontsize=13)
        plt.ylabel("True label", fontsize=13)
        if matched is None:
            plt.savefig(f"{self.path}/contingency_matrix.png")
        else:
            plt.savefig(f"{self.path}/contingency_matrix_{matched}.png")

    def matchLabelsInCM(self, cm, ground_truth_labels, clustered_labels):
        number_of_gtl = len(ground_truth_labels)
        number_of_cl = len(clustered_labels)

        match_labels = {}
        matched_labels = []
        converted_cm = [[] for _ in range(number_of_gtl)]

        for i, row in enumerate(cm):
            for elem in row:
                converted_cm[i].append(elem)
        print(converted_cm)

        if number_of_cl < number_of_gtl:
            for t in range(number_of_gtl - number_of_cl):
                for i, row in enumerate(converted_cm):
                    converted_cm[i].append(0)
                clustered_labels.append(number_of_cl+t)
        print(converted_cm)
        print(clustered_labels)

        copy_cm = copy.deepcopy(converted_cm)
        for _ in range(len(ground_truth_labels)):
            max_row_values = []
            indexes = []
            for array in copy_cm:
                max_row_values.append(max(array))
                indexes.append(np.argmax(array))
            gt_label = np.argmax(max_row_values)
            clustered_label = indexes[gt_label]
            match_labels[gt_label] = clustered_label
            matched_labels.append(clustered_label)

            for i in range(len(clustered_labels)):
                copy_cm[gt_label][i] = -1
            for i in range(len(ground_truth_labels)):
                copy_cm[i][clustered_label] = -1
            print(copy_cm)
            self.logger.info(copy_cm)
        print(f"Match_Labels: {match_labels}")
        self.logger.info(f"Match_Labels: {match_labels}")

        new_cm = [[] for _ in range(number_of_gtl)]
        new_clustered_labels = []

        for gt_label in ground_truth_labels:
            matched_clustered_label = match_labels[gt_label]
            for index, row in enumerate(converted_cm):
                new_cm[index].append(converted_cm[index][matched_clustered_label])
            new_clustered_labels.append(matched_clustered_label)

        for label in clustered_labels:
            if label not in matched_labels:
                for index, row in enumerate(converted_cm):
                    new_cm[index].append(converted_cm[index][label])
                new_clustered_labels.append(label)

        new_cm = np.asarray(new_cm)
        self.logger.info("New Contingency Matrix: ")
        self.logger.info(new_cm)
        print(f"New Contingency Matrix: ")
        print(new_cm)
        self.logger.info(f"New Clustered Label Sequence: {new_clustered_labels}")
        print(f"New Clustered Label Sequence: {new_clustered_labels}")
        return new_cm, new_clustered_labels

    def printMetrics(self, ground_truth, predictions):
        ground_truth_labels = sorted(np.unique(ground_truth))
        clustered_labels = sorted(np.unique(predictions))

        cm = contingency_matrix(ground_truth, predictions)
        self.logger.info("Contingency Matrix: ")
        self.logger.info(cm)
        print("Contingency Matrix: ")
        print(cm)
        self.printContingencyMatrix(cm, ground_truth_labels, clustered_labels)

        new_cm, new_clustered_labels = self.matchLabelsInCM(cm, ground_truth_labels, clustered_labels)
        self.printContingencyMatrix(new_cm, ground_truth_labels, new_clustered_labels, matched="matched")

        diagonal_elements = []
        for i in range(len(ground_truth_labels)):
            diagonal_elements.append(new_cm[i][i])
        sum_diagonal_elements = sum(diagonal_elements)
        self.logger.info(f"Diagonal_Elements: {diagonal_elements}, Sum: {sum_diagonal_elements}")
        print(f"Diagonal_Elements: {diagonal_elements}, Sum: {sum_diagonal_elements}")

        all_elements = []
        for row in new_cm:
            for elem in row:
                all_elements.append(elem)
        sum_all_elements = sum(all_elements)
        self.logger.info(f"All_Elements: {all_elements}, Sum: {sum_all_elements}")
        print(f"All_Elements: {all_elements}, Sum: {sum_all_elements}")

        accuracy = sum_diagonal_elements / sum_all_elements
        self.logger.info(f"Accuracy: {accuracy}")
        print(f"Accuracy: {accuracy}")
