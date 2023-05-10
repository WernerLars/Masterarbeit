import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import sort
from sklearn.metrics.cluster import contingency_matrix


class Visualisation(object):
    def __init__(self, variant_name, dataset_name, exp_path="", pc=""):
        self.variant_name = variant_name
        self.dataset_name = dataset_name
        self.logger = None
        self.timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")

        # Creating Path for figures
        if os.path.exists(f"{exp_path}{self.dataset_name[1]}") is False:
            os.mkdir(f"{exp_path}{self.dataset_name[1]}")
        if os.path.exists(f"{exp_path}{self.dataset_name[1]}/{self.variant_name}") is False:
            os.mkdir(f"{exp_path}{self.dataset_name[1]}/{self.variant_name}")
        if pc is not "":
            self.path = f"{exp_path}{self.dataset_name[1]}/{self.variant_name}/{pc}"
        else:
            self.path = f"{exp_path}{self.dataset_name[1]}/{self.variant_name}/{self.timestamp}"
        os.mkdir(self.path)

    def get_visualisation_path(self):
        return self.path

    def set_logger(self, logger):
        self.logger = logger

    def print_spike(self, spike, n_features, color, filename):
        """
            creates and saves a figure which shows the graph of a spike
        """

        plt.figure(figsize=(4, 4))
        t = np.arange(0, n_features, 1)
        plt.plot(t, spike, color=color)
        plt.savefig(f"{self.path}/{filename}.png")
        plt.close()

    def visualising_reconstructed_spike(self, original, reconstructed, n_features, cluster):
        """
            creates and saves a figure which shows the graph of a spike and its reconstruction
        """

        plt.figure(figsize=(4, 4))
        t = np.arange(0, n_features, 1)
        title = "True Label: " + cluster
        plt.plot(t, original, label="original")
        plt.plot(t, reconstructed, label="reconstructed")
        plt.legend(loc="upper left")
        plt.title(title)
        plt.savefig(f"{self.path}/reconstructedSpike_cluster_{cluster}.png")
        plt.close()

    def get_cluster_centers(self, features_list, labels):
        """
            computes cluster centers for x and y dimension of a feature list
            computation of center by using mean over all x and y values respectively
                only x and y values of a specific cluster are used
            :return: centroids as a list (every entry is a list with x and y coordinates
                     that represents a centroid of a cluster)
        """

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

    def visualising_features(self, x, y, filename=""):
        """
           creates and saves a figure of the feature space as a scatter plot
        """

        plt.figure(figsize=(8, 8))
        plt.scatter(x, y, color="k")
        plt.savefig(f"{self.path}/clusters_features{filename}.png")
        plt.close()

    def visualising_clusters(self, x, y, cluster, centroids=None, filename=""):
        """
            creates and saves a figure of the clusters in a feature space as scatter plots
            if centroids are passed they are marked as + for a cluster center
        """

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
        plt.close()

    def print_loss_curve(self, loss_values, epoch=""):
        """
           creates and saves loss curve figures for a list of loss values
        """

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
        plt.close()

    def print_contingency_matrix(self, cm, true_labels, predicted_labels, matched=None):
        """
            creating and saving contingency matrix figure with imshow (cm is numpy array)
            label color of text is changed from yellow to black if its value is >= 50
            x-axis are predicted labels, y-axis are true labels
        """

        plt.figure(figsize=(12, 6))
        plt.imshow(cm, interpolation="nearest")
        for (j, i), label in np.ndenumerate(cm):
            if label < 50:
                color = "yellow"
            else:
                color = "black"
            plt.text(i, j, label, ha="center", va="center", fontsize=13, color=color)
        plt.xticks(ticks=np.arange(len(predicted_labels)), labels=predicted_labels, fontsize=13)
        plt.yticks(true_labels, fontsize=13)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=13)
        plt.xlabel("Predicted label", fontsize=13)
        plt.ylabel("True label", fontsize=13)
        if matched is None:
            plt.title("Contingency Matrix", fontsize=16)
            plt.savefig(f"{self.path}/contingency_matrix.png")
        else:
            plt.title("Matched Contingency Matrix", fontsize=16)
            plt.savefig(f"{self.path}/contingency_matrix_{matched}.png")
        plt.close()

    def match_labels_in_cm(self, cm, ground_truth_labels, clustered_labels):
        """
            match ground truth labels with clustered labels
                creating deep copy of cm to be able to change values to -1
                -1 for finding the best match of labels
            add columns to a new contingency matrix with respect to matched labels
                first the matched labels, then not matches clusters
        :return: new contingency matrix, new clustered labels (for printing cm)
        """
        
        number_of_gtl = len(ground_truth_labels)
        number_of_cl = len(clustered_labels)

        match_labels = {}
        matched_labels = []
        converted_cm = [[] for _ in range(number_of_gtl)]

        # Convert Numpy Array into List by appending every element of cm to converted_cm
        for i, row in enumerate(cm):
            for elem in row:
                converted_cm[i].append(elem)
        self.logger.info(converted_cm)

        # If Number of Clusters is lower than the number of Ground Truth Labels,
        # then the Contingency Matrix needs to be filled up to have an n*n array
        if number_of_cl < number_of_gtl:
            for t in range(number_of_gtl - number_of_cl):
                for i, row in enumerate(converted_cm):
                    converted_cm[i].append(0)
                clustered_labels.append(number_of_cl+t)
        self.logger.info(converted_cm)
        self.logger.info(clustered_labels)

        # Deep Copy of cm for later to swap columns (copy_cm rows and columns get set to -1
        # for finding the best match of labels)
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

            # Replacing matched labels column and row values with -1
            for i in range(len(clustered_labels)):
                copy_cm[gt_label][i] = -1
            for i in range(len(ground_truth_labels)):
                copy_cm[i][clustered_label] = -1
            self.logger.info(copy_cm)

        self.logger.info(f"Match_Labels: {match_labels}")

        # new cm contains swapped columns where diagonal has matched labels
        new_cm = [[] for _ in range(number_of_gtl)]

        # new_clustered_labels only for the x-axis sequence in visualisation of contingency matrix
        new_clustered_labels = []

        # Adding the best columns to new_cm
        for gt_label in ground_truth_labels:
            matched_clustered_label = match_labels[gt_label]
            for index, row in enumerate(converted_cm):
                new_cm[index].append(converted_cm[index][matched_clustered_label])
            new_clustered_labels.append(matched_clustered_label)

        # Adding left over columns
        for label in clustered_labels:
            if label not in matched_labels:
                for index, row in enumerate(converted_cm):
                    new_cm[index].append(converted_cm[index][label])
                new_clustered_labels.append(label)

        # Convert new_cm to Numpy Array
        new_cm = np.asarray(new_cm)
        self.logger.info("New Contingency Matrix: ")
        self.logger.info(new_cm)
        self.logger.info(f"New Clustered Label Sequence: {new_clustered_labels}")
        return new_cm, new_clustered_labels

    def print_metrics(self, ground_truth, predictions):
        """
            creates contingency matrices before and after matching labels
            computes accuracy metric after matching labels with formula
                accuracy = TP+TN / TP+TN+FP+FN = sum diagonal elements / sum all elements
        """

        # Lists of sorted unique ground truth and clustered labels
        ground_truth_labels = sorted(np.unique(ground_truth))
        clustered_labels = sorted(np.unique(predictions))

        # Printing Contingency Matrix without matching labels
        cm = contingency_matrix(ground_truth, predictions)
        self.logger.info("Contingency Matrix: ")
        self.logger.info(cm)
        self.print_contingency_matrix(cm, ground_truth_labels, clustered_labels)

        # Matching Labels in Contingency Matrix
        new_cm, new_clustered_labels = self.match_labels_in_cm(cm, ground_truth_labels, clustered_labels)

        # Printing Contingency Matrix with matching labels
        self.print_contingency_matrix(new_cm, ground_truth_labels, new_clustered_labels, matched="matched")

        # Sum up all diagonal elements to get all TP and TN
        diagonal_elements = []
        for i in range(len(ground_truth_labels)):
            diagonal_elements.append(new_cm[i][i])
        sum_diagonal_elements = sum(diagonal_elements)
        self.logger.info(f"Diagonal_Elements: {diagonal_elements}, Sum: {sum_diagonal_elements}")

        # Sum up all elements (TP+TN+FP+FN)
        all_elements = []
        for row in new_cm:
            for elem in row:
                all_elements.append(elem)
        sum_all_elements = sum(all_elements)
        self.logger.info(f"All_Elements: {all_elements}, Sum: {sum_all_elements}")

        # Compute Accuracy with Formula
        accuracy = sum_diagonal_elements / sum_all_elements
        self.logger.info(f"Accuracy: {accuracy}")
