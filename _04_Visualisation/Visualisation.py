import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score


def visualisingClusters(x, y, cluster, centroids=None):
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
    plt.show()
    return


def printSpike(spike, n_features, color):
    plt.figure(figsize=(4, 4))
    t = np.arange(0, n_features, 1)
    plt.plot(t, spike, color=color)
    plt.show()
    return


def visualisingReconstructedSpike(original, reconstructed, n_features, cluster):
    plt.figure(figsize=(4, 4))
    t = np.arange(0, n_features, 1)
    title = "True Label: " + cluster
    plt.plot(t, original, label="original")
    plt.plot(t, reconstructed, label="reconstructed")
    plt.legend(loc="upper left")
    plt.title(title)
    plt.show()
    return


def printConfusionMatrix(ground_truth, predictions, labels):
    cm = confusion_matrix(ground_truth, predictions, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.title("Without Cluster Mapping")
    plt.show()

    mapping = []
    for true in labels:
        get_truth_values = []
        for predicted in labels:
            get_truth_values.append(cm[predicted][true])
        mapping.append(np.argmax(get_truth_values))

    predictions_mapping = []
    for prediction in predictions:
        predictions_mapping.append(mapping[prediction])

    cm_mapping = confusion_matrix(ground_truth, predictions_mapping, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_mapping, display_labels=labels)
    disp.plot()
    plt.title("With Cluster Mapping")
    plt.show()

    target_names = []
    for label in labels:
        target_names.append(f"cluster_{label}")

    print(f"Accuracy: {accuracy_score(ground_truth, predictions_mapping)}")
    cr = classification_report(ground_truth, predictions_mapping, target_names=target_names)
    print(cr)
