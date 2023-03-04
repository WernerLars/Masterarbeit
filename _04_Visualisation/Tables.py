import os
import textwrap

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


class Tables(object):
    def __init__(self):
        self.experiment_path = "../_05_Experiments/Base_Line_w_pc_0.6"
        self.filename = "informations.log"
        self.dataset_names = []
        self.experiment_names = []
        self.experiment_counter = 0
        self.variant_names = []
        self.accuracys = []
        self.punishment_coefficients = {}
        self.df = None

    def getInformationFromLog(self, log_path):
        with open(log_path) as log:
            accuracy = 0
            punishment_coefficient = None
            data_index = 0
            for line in log.readlines():
                if line.startswith("Experiment_path:"):
                    split = line.split(":")
                    if split[1] not in self.experiment_names:
                        self.experiment_counter += 1
                        self.experiment_names.append(split[1])
                        print(self.experiment_names)
                elif line.startswith("Accuracy:"):
                    split = line.split(":")
                    accuracy = round(float(split[1]), 4)
                elif line.startswith("Punishment_Coefficient:"):
                    split = line.split(":")
                    punishment_coefficient = split[1]
                elif line.startswith("Variant_name:"):
                    split = line.split(":")
                    variant_name = split[1][12:-1]
                    if f"{self.experiment_counter}_{variant_name}" not in self.variant_names:
                        self.variant_names.append(f"{self.experiment_counter}_{variant_name}")
                        self.punishment_coefficients[variant_name] = []
                        print(self.variant_names)
                elif line.startswith("Dataset_Path:"):
                    split = line.split(":")
                    dataset_name = split[1][16:].split("/")[2][:-5]
                    if dataset_name not in self.dataset_names:
                        self.dataset_names.append(dataset_name)
                        self.accuracys.append([])
                    data_index = self.dataset_names.index(dataset_name)
            self.accuracys[data_index].append(round(accuracy * 100, 4))
            self.punishment_coefficients[variant_name].append(punishment_coefficient)

    def printAccuracyTable(self):
        plt.figure(figsize=(5 + 2 * len(self.variant_names), 6))
        ax = sns.heatmap(self.df, cmap="Spectral", vmin=0, vmax=100, annot=True, fmt=".2f")
        for t in ax.texts: t.set_text(t.get_text() + " %")
        s = np.arange(len(self.variant_names)) + 0.5
        plt.xticks(s, [textwrap.fill(x.replace('_', ' '), width=20) for x in self.variant_names], rotation=0)
        plt.tight_layout()
        plt.savefig(f"{self.experiment_path}/AccuracyTable.png")
        # df.to_csv(f"{self.experiment_path}/AccuracyTable.csv")
        # print(df.to_latex())

    def printAccuracyGraph(self):
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

    def printBoxplot1(self):
        plt.figure(figsize=(10 + 1.7 * len(self.dataset_names), 20))
        for i, variant in enumerate(self.variant_names):
            graph = self.df[variant]
            plt.scatter(self.dataset_names, graph, label=variant, marker="*", s=500)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
        plt.yticks(fontsize=23)
        plt.xticks(np.arange(len(self.dataset_names)), [x.replace('_', '\n') for x in self.dataset_names],
                   fontsize=20)
        plt.legend(loc="lower right", fontsize=23)
        plt.title("Accuracy for Datasets", fontsize=30)
        plt.tight_layout()
        plt.savefig(f"{self.experiment_path}/Boxplot1.png")

    def printBoxplot2(self):
        plt.figure(figsize=(15 + 1.7 * len(self.variant_names), 20))
        for i, dataset in enumerate(self.dataset_names):
            t = np.arange(len(self.accuracys[i]))
            plt.scatter(t, self.accuracys[i], label=dataset, marker="*", s=500)
        plt.title(f"Accuracy for Variants", fontsize=30)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
        plt.yticks(fontsize=23)
        plt.xticks(np.arange(len(self.accuracys[0])), [x.replace('_', '\n') for x in self.variant_names], fontsize=23)
        plt.xlim(-0.5, len(self.variant_names))
        plt.legend(loc="center left", fontsize=23, bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f"{self.experiment_path}/Boxplot2.png")


def main():
    tables = Tables()
    for (root, dirs, files) in os.walk(tables.experiment_path):
        for file in files:
            if file == tables.filename:
                log_path = f"{root}\{file}"
                tables.getInformationFromLog(log_path)
    tables.df = pd.DataFrame(tables.accuracys,
                             index=tables.dataset_names,
                             columns=tables.variant_names)
    print(tables.accuracys)
    print(tables.dataset_names)
    print(tables.variant_names)
    tables.printAccuracyGraph()
    tables.printBoxplot1()
    tables.printBoxplot2()
    tables.df.loc['mean'] = tables.df.mean()
    tables.printAccuracyTable()

    pc = pd.DataFrame.from_dict(tables.punishment_coefficients)
    pc.set_axis(tables.dataset_names, axis='index', inplace=True)
    pc.to_latex(f"{tables.experiment_path}/Punishment_Coefficients.txt")


if __name__ == '__main__':
    main()
