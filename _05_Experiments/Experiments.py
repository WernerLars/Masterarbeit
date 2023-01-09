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

    path = datasets[1]
    dataset_name = path[16:].split("/")
    variant = variants[2]
    vis = Visualisation(variant, dataset_name)
    vispath = vis.getVisualisationPath()

    logging.basicConfig(filename=f"{vispath}/informations.log",
                        format="%(message)s",
                        filemode="w")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info(f"Path: {path}")
    logger.info(f"Dataset_name: {dataset_name}")
    logger.info(f"Variant: {variant}")

    if variant == "Variant_01_PCA_KMeans":
        Variant_01_PCA_KMeans(path, vis, logger)
    elif variant == "Variant_02_Autoencoder_KMeans":
        Variant_02_Autoencoder_KMeans(path, vis, logger)
    elif variant == "Variant_03_PCA_QLearning":
        Variant_03_PCA_QLearning(path, vis, logger)
    elif variant == "Variant_04_Offline_Autoencoder_QLearning":
        Variant_04_Offline_Autoencoder_QLearning(path, vis, logger)
    elif variant == "Variant_05_Online_Autoencoder_QLearning":
        Variant_05_Online_Autoencoder_QLearning(path, vis, logger)


if __name__ == '__main__':
    main()
