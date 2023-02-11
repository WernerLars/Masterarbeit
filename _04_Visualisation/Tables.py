import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class Tables(object):
    def __init__(self):
        self.experiment_path = "../_05_Experiments/Experiment_01_2023_02_10-11_24_37"
        self.filename = "informations.log"
        self.dataset_names = []
        self.variant_names = []
        self.accuracys = []

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
        df = pd.DataFrame(self.accuracys,
                          index=self.dataset_names,
                          columns=self.variant_names)
        plt.figure(figsize=(20, 6))
        sns.heatmap(df, cmap="Greens", annot=True)
        plt.xticks(rotation=0)
        plt.savefig(f"{self.experiment_path}/AccuracyTable.png")
        #df.to_csv(f"{self.experiment_path}/AccuracyTable.csv")
        #print(df.to_latex())

    def printAccuracyGraph(self):
        return


def main():
    tables = Tables()
    for (root, dirs, files) in os.walk(tables.experiment_path):
        for file in files:
            if file == tables.filename:
                log_path = f"{root}\{file}"
                tables.getInformationFromLog(log_path)
    print(tables.accuracys)
    print(tables.dataset_names)
    print(tables.variant_names)
    tables.printAccuracyTable()


if __name__ == '__main__':
    main()
