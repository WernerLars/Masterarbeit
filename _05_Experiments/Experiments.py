from _03_SpikeSorter.Variant_01_PCA_KMeans import Variant_01_PCA_KMeans
from _03_SpikeSorter.Variant_02_Autoencoder_KMeans import Variant_02_Autoencoder_KMeans
from _03_SpikeSorter.Variant_03_PCA_QLearning import Variant_03_PCA_QLearning
from _03_SpikeSorter.Variant_04_Offline_Autoencoder_QLearning import Variant_04_Offline_Autoencoder_QLearning
from _03_SpikeSorter.Variant_05_Online_Autoencoder_QLearning import Variant_05_Online_Autoencoder_QLearning
from _04_Visualisation.Visualisation import Visualisation
import logging


def main():
    datasets = {
        1: "../_00_Datasets/01_SimDaten_Martinez2009/simulation_1.mat",
        2: "../_00_Datasets/03_SimDaten_Quiroga2020/004_C_Difficult1_noise005.mat",
        3: "../_00_Datasets/03_SimDaten_Quiroga2020/016_C_Easy1_noise005.mat"
    }
    variants = {
        1: "Variant_01_PCA_KMeans",
        2: "Variant_02_Autoencoder_KMeans",
        3: "Variant_03_PCA_QLearning",
        4: "Variant_04_Offline_Autoencoder_QLearning",
        5: "Variant_05_Online_Autoencoder_QLearning"
    }

    path = datasets[3]
    dataset_name = path[16:].split("/")
    variant_name = variants[5]
    vis = Visualisation(variant_name, dataset_name)
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


if __name__ == '__main__':
    main()
