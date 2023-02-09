import os
import time

from _03_SpikeSorter.Variant_01_PCA_KMeans import Variant_01_PCA_KMeans
from _03_SpikeSorter.Variant_02_Autoencoder_KMeans import Variant_02_Autoencoder_KMeans
from _03_SpikeSorter.Variant_03_PCA_QLearning import Variant_03_PCA_QLearning
from _03_SpikeSorter.Variant_04_Offline_Autoencoder_QLearning import Variant_04_Offline_Autoencoder_QLearning
from _03_SpikeSorter.Variant_05_Online_Autoencoder_QLearning import Variant_05_Online_Autoencoder_QLearning
from _04_Visualisation.Visualisation import Visualisation
import logging


def main():
    datasets = {
        1: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise005.mat",
        # 2: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise010.mat",
        # 3: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise015.mat",
        # 4: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise020.mat",
        # 5: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise025.mat",
        # 6: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise030.mat",
        # 7: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise035.mat",
        # 8: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise040.mat",
        # 9: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy2_noise005.mat",
        # 10: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy2_noise010.mat",
        # 11: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy2_noise015.mat",
        # 12: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy2_noise020.mat",
        # 13: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult1_noise005.mat",
        # 14: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult1_noise010.mat",
        # 15: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult1_noise015.mat",
        # 16: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult1_noise020.mat",
        # 17: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult2_noise005.mat",
        # 18: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult2_noise010.mat",
        # 19: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult2_noise015.mat",
        # 20: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult2_noise020.mat",
        # 21: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Burst_Easy2_noise015.mat",
        # 22: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Drift_Easy2_noise015.mat",
    }
    variants = {
        1: "Variant_01_PCA_KMeans",
        2: "Variant_02_Autoencoder_KMeans",
        # 3: "Variant_03_PCA_QLearning",
        # 4: "Variant_04_Offline_Autoencoder_QLearning",
        # 5: "Variant_05_Online_Autoencoder_QLearning"
    }

    exp_name = "Experiment_01"
    timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    exp_path = f"{exp_name}_{timestamp}"
    os.mkdir(exp_path)

    for variant in variants:
        for dataset in datasets:

            print(variants[variant])
            print(datasets[dataset])

            path = datasets[dataset]
            dataset_name = path[16:].split("/")
            variant_name = variants[variant]
            vis = Visualisation(variant_name, dataset_name, exp_path=f"{exp_path}/")
            vis_path = vis.getVisualisationPath()

            formatter = logging.Formatter("%(message)s")
            handler1 = logging.FileHandler(filename=f"{vis_path}/informations.log", mode="w")
            handler1.setFormatter(formatter)
            logger = logging.getLogger("Information Logger")
            logger.setLevel(logging.INFO)
            logger.addHandler(handler1)

            vis.setLogger(logger)

            handler2 = logging.FileHandler(filename=f"{vis_path}/parameters.log", mode="w")
            handler2.setFormatter(formatter)
            parameter_logger = logging.getLogger("Parameter Logger")
            parameter_logger.setLevel(logging.INFO)
            parameter_logger.addHandler(handler2)

            logger.info(f"Dataset_Path: {path}")
            parameter_logger.info(f"Dataset_Path: {path}")
            logger.info(f"Dataset_name: {dataset_name}")
            parameter_logger.info(f"Dataset_name: {dataset_name}")
            logger.info(f"Variant_name: {variant_name}")
            parameter_logger.info(f"Variant_name: {variant_name}")
            logger.info(f"Visualisation_Path: {vis_path}")
            parameter_logger.info(f"Visualisation_Path: {vis_path}")

            if variant_name == "Variant_01_PCA_KMeans":
                Variant_01_PCA_KMeans(path, vis, logger, parameter_logger)
            elif variant_name == "Variant_02_Autoencoder_KMeans":
                Variant_02_Autoencoder_KMeans(path, vis, logger, parameter_logger)
            elif variant_name == "Variant_03_PCA_QLearning":
                Variant_03_PCA_QLearning(path, vis, logger, parameter_logger)
            elif variant_name == "Variant_04_Offline_Autoencoder_QLearning":
                Variant_04_Offline_Autoencoder_QLearning(path, vis, logger, parameter_logger)
            elif variant_name == "Variant_05_Online_Autoencoder_QLearning":
                Variant_05_Online_Autoencoder_QLearning(path, vis, logger, parameter_logger)

            handler1.close()
            handler2.close()
            logger.removeHandler(handler1)
            parameter_logger.removeHandler(handler2)

if __name__ == '__main__':
    main()
