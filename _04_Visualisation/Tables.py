import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


class Tables(object):
    def __init__(self):
        self.experiment_path = "../_05_Experiments/AE_Model_1/Base_Line"
        self.filename = "informations.log"
        self.dataset_names = []
        self.experiment_names = []
        self.experiment_counter = 0
        self.variant_names = []
        self.accuracys = []
        self.punishment_coefficients = {}
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
                    punishment_coefficient = split[1]

                # Extract Variant Name
                elif line.startswith("Variant_name:"):
                    split = line.split(":")
                    variant_name = split[1][12:-1]

                    # Check if experiment and variant combination is not in variant name list
                    if f"{self.experiment_counter}_{variant_name}" not in self.variant_names:
                        self.variant_names.append(f"{self.experiment_counter}_{variant_name}")

                        # Initialise new variant key in punishment coefficient dictionary
                        self.punishment_coefficients[variant_name] = []

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
            self.accuracys[data_index].append(round(accuracy * 100, 3))
            self.punishment_coefficients[variant_name].append(punishment_coefficient)

    def print_accuracy_table(self, random_seeds=False, std=None):
        """
            creates and saves a figure of a table, which is created with seaborn
            table has variant names on x-axis and dataset names on y-axis and contains
                accuracy values of variant dataset combination
                column and row of mean values for dataset and variant
                standard derivation column
        """

        if random_seeds:
            plt.figure(figsize=(1.3 * len(self.variant_names), 7))
        else:
            plt.figure(figsize=(2 + 1.4 * len(self.variant_names), 6))
        ax = sns.heatmap(self.df, cmap="Spectral", vmin=0, vmax=100, annot=True, fmt=".1f", linewidths=0.5)
        std_counter = 0
        for index, t in enumerate(ax.texts):
            if (index % len(self.variant_names) == len(self.variant_names) - 1) and random_seeds:
                t.set_text(t.get_text() + " %: " + str(round(std[std_counter], 1)))
                std_counter += 1
            else:
                t.set_text(t.get_text() + " %")

        y_names = [x.replace('_', ' ') for x in self.df.index]
        y_names = [x.replace('C', '') for x in y_names]
        y_names = [x.replace('Difficult', 'D') for x in y_names]
        y_names = [x.replace('Easy', 'E') for x in y_names]
        y_names = [x.replace('noise', '') for x in y_names]
        y_names = [x.replace('Burst', 'B') for x in y_names]
        y_names = [x.replace('Drift', 'D') for x in y_names]
        s = np.arange(len(self.dataset_names)+1) + 0.5
        plt.yticks(s, y_names)

        s = np.arange(len(self.variant_names))
        if random_seeds:
            x_names = [str(x) for x in np.arange(len(self.variant_names)-1)]
            x_names.append("mean:std")
            plt.xticks(s+0.5, x_names, rotation=0)
            plt.title(f"Random Seeds von {self.variant_names[0][2:]}")
        else:
            s = np.arange(len(self.variant_names))
            plt.xticks(s+0.5, [x.replace('_', '\n') for x in self.variant_names], rotation=0)
            plt.title(f"Mean Accuracy: {round(self.df.loc['mean'].mean(), 2)} %")
        plt.tight_layout()
        plt.savefig(f"{self.experiment_path}/AccuracyTable.png")
        plt.close()

    def print_accuracy_graph(self):
        """
            creates and saves bar figures for every variant in variant names
            on x-axis are datasets and y-axis are accuracy values
            every bar represents accuracy value of the dataset for the specific variant
        """

        for i, variant in enumerate(self.variant_names):
            plt.figure(figsize=(10 + 1.7 * len(self.dataset_names), 15))
            graph = self.df[variant]
            t = np.arange(len(graph))
            plt.bar(t, self.df[variant], label=variant, align="center")
            for j, y in enumerate(graph):
                plt.text(t[j], graph[j] - 5, f"{graph[j]}%", ha='center', color="w", fontsize=23)
            plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
            plt.yticks(fontsize=23)
            plt.xlim(-0.5, len(graph) - 0.5)
            plt.xticks(t, [x.replace('_', '\n') for x in self.dataset_names], rotation=0, fontsize=23)
            plt.title(f"Mean Accuracy Graph of {variant}: {np.mean(graph).round(2)}%", fontsize=30)
            plt.tight_layout()
            plt.savefig(f"{self.experiment_path}/AccuracyGraph_{variant}.png")
            plt.close()

    def print_violent_graph_datasets(self):
        """
            creates a violent graph, which represents accuracy over datasets
            x-axis are datasets and y-axis are accuracy values
        """

        plt.figure(figsize=(8 + 1 * len(self.dataset_names), 20))
        sns.violinplot(data=self.accuracys, cut=0, scale='width')
        plt.yticks(fontsize=30)
        x_names = [x.replace('_', '\n')[1:] for x in self.dataset_names]
        x_names = [x.replace('Difficult', 'D') for x in x_names]
        x_names = [x.replace('Easy', 'E') for x in x_names]
        x_names = [x.replace('noise', '') for x in x_names]
        x_names = [x.replace('Burst', 'B') for x in x_names]
        x_names = [x.replace('Drift', 'D') for x in x_names]

        plt.xticks(np.arange(len(self.dataset_names)), x_names,
                   fontsize=30)
        plt.title("Accuracy auf Datensätzen", fontsize=40)
        plt.tight_layout()
        plt.savefig(f"{self.experiment_path}/Violent_Plot_Datasets.png")
        plt.close()

    def print_violent_graphs_variants(self, random_seeds=False):
        """
            creates a violent graph, which represents accuracy over variants
            x-axis are variants and y-axis are accuracy values
        """

        plt.figure(figsize=(10 + 1.1 * len(self.variant_names), 9))
        sns.violinplot(data=self.df, cut=0, scale='width')
        plt.title(f"Accuracy der Varianten", fontsize=30)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
        plt.yticks(fontsize=23)

        s = np.arange(len(self.accuracys[0]))
        if random_seeds:
            plt.xticks(s, s, fontsize=23)
        else:
            plt.xticks(s, [x.replace('_', '\n') for x in self.variant_names], fontsize=23)
        plt.xlim(-0.5, len(self.variant_names))
        plt.tight_layout()
        plt.savefig(f"{self.experiment_path}/Violent_Plot_Variants.png")
        plt.close()


def main(experiment_path="", random_seeds=False):
    tables = Tables()

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
    tables.df = pd.DataFrame(tables.accuracys,
                             index=tables.dataset_names,
                             columns=tables.variant_names)

    # Printing Graphs for the experiments
    tables.print_accuracy_graph()
    tables.print_violent_graph_datasets()
    tables.print_violent_graphs_variants(random_seeds)

    # Adding mean and variance to dataframe
    # this is after above functions because dataframe
    # is used to create figures
    if random_seeds:
        tables.df.loc['mean'] = tables.df.mean(axis=0)
        tables.df['mean'] = tables.df.mean(axis=1)
        tables.variant_names.append("mean:std")
        std = tables.df.std(axis=1)
        tables.print_accuracy_table(random_seeds, std)
    else:
        tables.df.loc['mean'] = tables.df.mean(axis=0)
        tables.print_accuracy_table()

    # Creating a new dataframe for printing a table of punishment coefficients
    # x-axis are variants, y-axis are datasets
    pc = pd.DataFrame.from_dict(tables.punishment_coefficients)
    pc.set_axis(tables.dataset_names, axis='index', inplace=True)
    pc.to_latex(f"{tables.experiment_path}/Punishment_Coefficients.txt", escape=False)


if __name__ == '__main__':
    main()
