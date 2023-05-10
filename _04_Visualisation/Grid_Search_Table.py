import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class Tables(object):
    def __init__(self):
        self.experiment_path = "../_05_Experiments/AE_Model_2/Epochs_GS_PC/V5_20"
        self.filename = "informations.log"
        self.dataset_names = []
        self.experiment_names = []
        self.experiment_counter = 0
        self.variant_names = []
        self.accuracys = []
        self.punishment_coefficients = []
        self.epoch_list = None
        self.df = None

    def get_information_from_log(self, log_path):
        """
            opens a log file from log_path and filters out information about experiment path,
                accuracy values, punishment coefficients, variant name, dataset path
            experiment path to differentiate between experiments (creating new key for variant names,
                so multiple experiments of same variant can be on table)
            saving accuracy and punishment coefficients into list/dictionary
            data_index:
        """

        with open(log_path) as log:

            # Initialising Parameters for log File
            accuracy = 0
            punishment_coefficient = None

            # Data index is used as index for accuracy list
            data_index = 0

            # Reading over every Line of log File
            for line in log.readlines():

                # Extract experiment path
                if line.startswith("Experiment_path:"):
                    split = line.split(":")

                    # check if experiment path has changed
                    if split[1] not in self.experiment_names:
                        # if new experiment path a counter is incremented and name is added
                        self.experiment_counter += 1
                        self.experiment_names.append(split[1])

                # Extract Accuracy Value
                elif line.startswith("Accuracy:"):
                    split = line.split(":")
                    accuracy = round(float(split[1]), 4)

                # Extract Punishment Coefficient
                elif line.startswith("Punishment_Coefficient:"):
                    split = line.split(":")
                    punishment_coefficient = round(float(split[1][:-1]), 1)

                # Extract Variant Name
                elif line.startswith("Variant_name:"):
                    split = line.split(":")
                    variant_name = split[1][12:-1]

                    # Check if experiment and variant combination is not in variant name list
                    if f"{self.experiment_counter}_{variant_name}" not in self.variant_names:
                        self.variant_names.append(f"{self.experiment_counter}_{variant_name}")

                # Extract Dataset Name
                elif line.startswith("Dataset_Path:"):
                    split = line.split(":")
                    dataset_name = split[1][16:].split("/")[2][:-5]

                    # For every new dataset a list is added to accuracy list
                    if dataset_name not in self.dataset_names:
                        self.dataset_names.append(dataset_name)
                        self.accuracys.append([])

                    # data_index is used to find right list to include accuracy in accuracy list
                    data_index = self.dataset_names.index(dataset_name)

            # After scanning all lines of log file:
            #   adding found accuracy in percent to accuracy list
            #   adding found punishment coefficient to pc list (if not None is added)
            self.accuracys[data_index].append(round(accuracy * 100, 4))
            if punishment_coefficient not in self.punishment_coefficients:
                self.punishment_coefficients.append(punishment_coefficient)

    def print_accuracy_table(self):
        """
            creates and saves a figure of a table, which is created with seaborn
            table has punishment_coefficients on x-axis and dataset names on y-axis and contains
                accuracy values of variant dataset punishment coefficient combination
        """

        if self.epoch_list is None:
            plt.figure(figsize=(0.85 * len(self.punishment_coefficients), 7))
        else:
            plt.figure(figsize=(0.85 * len(self.epoch_list), 7))
        ax = sns.heatmap(self.df, cmap="Spectral", vmin=0, vmax=100, annot=True, fmt=".1f", linewidths=0.5)
        for t in ax.texts:
            t.set_text(t.get_text() + " %")

        y_names = [x.replace('_', ' ') for x in self.df.index]
        y_names = [x.replace('C', '') for x in y_names]
        y_names = [x.replace('Difficult', 'D') for x in y_names]
        y_names = [x.replace('Easy', 'E') for x in y_names]
        y_names = [x.replace('noise', '') for x in y_names]
        y_names = [x.replace('Burst', 'B') for x in y_names]
        y_names = [x.replace('Drift', 'D') for x in y_names]
        s = np.arange(len(self.dataset_names)) + 0.5
        plt.yticks(s, y_names)

        if self.epoch_list is None:
            s = np.arange(len(self.punishment_coefficients)) + 0.5
            plt.xticks(s, self.punishment_coefficients, rotation=0)
        else:
            s = np.arange(len(self.epoch_list)) + 0.5
            plt.xticks(s, self.epoch_list, rotation=0)
        plt.tight_layout()
        plt.savefig(f"{self.experiment_path}/AccuracyTable.png")
        plt.close()


def main(experiment_path="", epoch_list=None):
    tables = Tables()
    tables.epoch_list = epoch_list

    # setting a custom experiment path
    if experiment_path is not "":
        tables.experiment_path = experiment_path

    # scanning over all files in the specified experiment path
    for (root, dirs, files) in os.walk(tables.experiment_path):
        for file in files:
            if file == tables.filename:
                log_path = f"{root}\{file}"
                tables.get_information_from_log(log_path)

    # Creating dataframe for visualisation
    if epoch_list is None:
        tables.df = pd.DataFrame(tables.accuracys,
                                 index=tables.dataset_names,
                                 columns=tables.punishment_coefficients)
    else:
        tables.df = pd.DataFrame(tables.accuracys,
                                 index=tables.dataset_names,
                                 columns=epoch_list)

    tables.print_accuracy_table()

    # Creating a new dataframe for printing a table of punishment coefficients
    # x-axis is variant, y-axis are datasets
    if epoch_list is None:
        best_pc = pd.DataFrame(tables.df.idxmax(axis=1),
                               index=tables.dataset_names,
                               columns=["best_pc"])

        best_pc.to_latex(f"{tables.experiment_path}/Best_Punishment_Coefficients.txt")


if __name__ == '__main__':
    main()
