import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


class Tables(object):
    def __init__(self):
        self.experiment_path = "../_05_Experiments/Test"
        self.filename = "informations.log"
        self.dataset_names = []
        self.variant_names = []
        self.accuracys = []
        self.df = None

    def getInformationFromLog(self, log_path):
        with open(log_path) as log:
            accuracy = 0
            data_index = 0
            for line in log.readlines():
                if line.startswith("Accuracy:"):
                    split = line.split(":")
                    accuracy = round(float(split[1]), 4)
                elif line.startswith("Variant_name:"):
                    split = line.split(":")
                    variant_name = split[1][12:-1]
                    if variant_name not in self.variant_names:
                        self.variant_names.append(variant_name)
                elif line.startswith("Dataset_Path:"):
                    split = line.split(":")
                    dataset_name = split[1][16:].split("/")[2][:-5]
                    if dataset_name not in self.dataset_names:
                        self.dataset_names.append(dataset_name)
                        self.accuracys.append([])
                    data_index = self.dataset_names.index(dataset_name)
            self.accuracys[data_index].append(accuracy)

    def printAccuracyTable(self):
        plt.figure(figsize=(20, 6))
        sns.heatmap(self.df, cmap="Spectral", annot=True)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{self.experiment_path}/AccuracyTable.png")
        # df.to_csv(f"{self.experiment_path}/AccuracyTable.csv")
        # print(df.to_latex())

    def printAccuracyGraph(self):
        for i, variant in enumerate(self.variant_names):
            plt.figure(figsize=(45, 15))
            graph = self.df[variant]
            t = np.arange(len(graph))
            plt.bar(t, self.df[variant], label=variant, align="center")
            for j, y in enumerate(graph):
                plt.text(t[j], graph[j] - 0.05, round(graph[j] * 100, 2), ha='center', color="w", fontsize=23)
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
            plt.yticks(fontsize=23)
            plt.xlim(-0.5, len(graph) - 0.5)
            plt.xticks(t, [x.replace('_', '\n') for x in self.dataset_names], rotation=0, fontsize=23)
            plt.title(f"Mean Accuracy Graph of {variant}: {round(np.mean(graph)*100, 2)}%", fontsize=30)
            plt.tight_layout()
            plt.savefig(f"{self.experiment_path}/AccuracyGraph_{variant}.png")

    def printBoxplot1(self):
        plt.figure(figsize=(40, 20))
        for i, variant in enumerate(self.variant_names):
            graph = self.df[variant]
            plt.scatter(self.dataset_names, graph, label=variant, marker="*", s=500)
        plt.yticks(fontsize=23)
        plt.xticks(np.arange(len(self.dataset_names)), [x.replace('_', '\n') for x in self.dataset_names],
                   fontsize=20)
        plt.legend(loc="lower right", fontsize=23)
        plt.title("Accuracy Graph", fontsize=30)
        plt.tight_layout()
        plt.savefig(f"{self.experiment_path}/Boxplot1.png")

    def printBoxplot2(self):
        plt.figure(figsize=(40, 20))
        for i, dataset in enumerate(self.dataset_names):
            t = np.arange(len(self.accuracys[i]))
            plt.scatter(t, self.accuracys[i], label=dataset, marker="*", s=500)
        plt.title(f"Boxplot", fontsize=30)
        plt.yticks(fontsize=23)
        plt.xticks(np.arange(len(self.accuracys[0])), [x.replace('_', '\n') for x in self.variant_names], fontsize=23)
        plt.xlim(-0.5, len(self.variant_names))
        plt.legend(loc="best", fontsize=23)
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


if __name__ == '__main__':
    main()
