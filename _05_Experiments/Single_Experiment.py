import random
import numpy as np
import torch

from _03_SpikeSorter.Variant_01_PCA_KMeans import Variant_01_PCA_KMeans
from _03_SpikeSorter.Variant_02_Autoencoder_KMeans import Variant_02_Autoencoder_KMeans
from _03_SpikeSorter.Variant_03_PCA_QLearning import Variant_03_PCA_QLearning
from _03_SpikeSorter.Variant_04_Offline_Autoencoder_QLearning import Variant_04_Offline_Autoencoder_QLearning
from _03_SpikeSorter.Variant_05_Online_Autoencoder_QLearning import Variant_05_Online_Autoencoder_QLearning
from _04_Visualisation.Visualisation import Visualisation
import logging


def main(main_path="", dataset=1, variant=1, pc=1, disable_tqdm=False, chooseAutoencoder=1, epoch=8):

    datasets = {
        1: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Burst_Easy2_noise015.mat",
        2: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult1_noise005.mat",
        3: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult1_noise010.mat",
        4: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult1_noise015.mat",
        5: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult1_noise020.mat",
        6: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult2_noise005.mat",
        7: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult2_noise010.mat",
        8: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult2_noise015.mat",
        9: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult2_noise020.mat",
        10: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Drift_Easy2_noise015.mat",
        11: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise005.mat",
        12: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise010.mat",
        13: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise015.mat",
        14: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise020.mat",
        15: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise025.mat",
        16: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise030.mat",
        17: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise035.mat",
        18: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise040.mat",
        19: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy2_noise005.mat",
        20: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy2_noise010.mat",
        21: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy2_noise015.mat",
        22: "../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy2_noise020.mat",
    }

    variants = {
        1: "Variant_01_PCA_KMeans",
        2: "Variant_02_Autoencoder_KMeans",
        3: "Variant_03_PCA_QLearning",
        4: "Variant_04_Offline_Autoencoder_QLearning",
        5: "Variant_05_Online_Autoencoder_QLearning",
        6: "Variant_05_Online_Autoencoder_QLearning_opt",
        7: "Variant_05_Online_Autoencoder_QLearning_opt_temp",
        8: "Variant_05_Online_Autoencoder_QLearning_opt_temp_noisy",

    }

    dataset_number = dataset
    variant_number = variant
    punishment_coefficient = pc

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    path = datasets[dataset_number]
    dataset_name = path[16:].split("/")
    variant_name = variants[variant_number]
    vis = Visualisation(variant_name, dataset_name, exp_path=main_path, pc=f"{punishment_coefficient}")
    vis_path = vis.get_visualisation_path()
    exp_path = f"Experiment_0{variant_number}"

    formatter = logging.Formatter("%(message)s")
    handler1 = logging.FileHandler(filename=f"{vis_path}/informations.log", mode="w")
    handler1.setFormatter(formatter)
    logger = logging.getLogger("Information Logger")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler1)

    vis.set_logger(logger)

    handler2 = logging.FileHandler(filename=f"{vis_path}/parameters.log", mode="w")
    handler2.setFormatter(formatter)
    parameter_logger = logging.getLogger("Parameter Logger")
    parameter_logger.setLevel(logging.INFO)
    parameter_logger.addHandler(handler2)

    parameter_logger.info(f"Seed: {seed}")
    logger.info(f"Experiment_path: {exp_path}")
    parameter_logger.info(f"Experiment_path: {exp_path}")
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
        Variant_02_Autoencoder_KMeans(path, vis, logger, parameter_logger,
                                      disable_tqdm=disable_tqdm,
                                      chooseAutoencoder=chooseAutoencoder,
                                      epochs=epochs)
    elif variant_name == "Variant_03_PCA_QLearning":
        Variant_03_PCA_QLearning(path, vis, logger, parameter_logger,
                                 punishment_coefficient=punishment_coefficient,
                                 q_learning_size=300,
                                 normalise=False,
                                 disable_tqdm=disable_tqdm)
    elif variant_name == "Variant_04_Offline_Autoencoder_QLearning":
        Variant_04_Offline_Autoencoder_QLearning(path, vis, logger, parameter_logger,
                                                 punishment_coefficient=punishment_coefficient,
                                                 normalise=False,
                                                 disable_tqdm=disable_tqdm,
                                                 chooseAutoencoder=chooseAutoencoder)
    elif variant_name == "Variant_05_Online_Autoencoder_QLearning":
        Variant_05_Online_Autoencoder_QLearning(path, vis, logger, parameter_logger,
                                                punishment_coefficient=punishment_coefficient,
                                                optimising=False,
                                                templateMatching=False,
                                                noisyBatch=False,
                                                normalise=False,
                                                disable_tqdm=disable_tqdm,
                                                chooseAutoencoder=chooseAutoencoder,
                                                epochs=epochs)
    elif variant_name == "Variant_05_Online_Autoencoder_QLearning_opt":
        Variant_05_Online_Autoencoder_QLearning(path, vis, logger, parameter_logger,
                                                punishment_coefficient=punishment_coefficient,
                                                optimising=True,
                                                templateMatching=False,
                                                noisyBatch=False,
                                                normalise=False,
                                                disable_tqdm=disable_tqdm,
                                                chooseAutoencoder=chooseAutoencoder,
                                                epochs=epochs)
    elif variant_name == "Variant_05_Online_Autoencoder_QLearning_opt_temp":
        Variant_05_Online_Autoencoder_QLearning(path, vis, logger, parameter_logger,
                                                punishment_coefficient=punishment_coefficient,
                                                optimising=True,
                                                templateMatching=True,
                                                noisyBatch=False,
                                                normalise=False,
                                                disable_tqdm=disable_tqdm,
                                                chooseAutoencoder=chooseAutoencoder,
                                                epochs=epochs)
    elif variant_name == "Variant_05_Online_Autoencoder_QLearning_opt_temp_noisy":
        Variant_05_Online_Autoencoder_QLearning(path, vis, logger, parameter_logger,
                                                punishment_coefficient=punishment_coefficient,
                                                optimising=True,
                                                templateMatching=True,
                                                noisyBatch=True,
                                                normalise=False,
                                                disable_tqdm=disable_tqdm,
                                                chooseAutoencoder=chooseAutoencoder,
                                                epochs=epochs)


if __name__ == '__main__':
    main()
