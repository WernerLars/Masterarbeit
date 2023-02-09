import os


class Tables(object):
    def __init__(self):
        self.experiment_path = "../_05_Experiments/Experiment_01_2023_02_09-15_05_55"
        self.filename = "informations.log"
        self.dataset_names = []
        self.variant_names = []
        self.accuracys = []

    def getInformationFromLog(self, log_path):
        with open(log_path) as log:
            for line in log.readlines():
                if line.startswith("Accuracy:"):
                    split = line.split(":")
                    self.accuracys.append(round(float(split[1]), 4))
                elif line.startswith("Variant_name:"):
                    split = line.split(":")
                    if split[1] not in self.variant_names:
                        self.variant_names.append(split[1][12:-2])
                elif line.startswith("Dataset_Path:"):
                    split = line.split(":")
                    dataset_name = split[1][16:].split("/")
                    if dataset_name[2] not in self.dataset_names:
                        self.dataset_names.append(dataset_name[2][:-5])

    def printAccuracyTable(self):
        return

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


if __name__ == '__main__':
    main()
