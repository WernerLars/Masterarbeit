import os
import random

import numpy as np
import torch

from _03_SpikeSorter.Variant_05_Online_Autoencoder_QLearning import Variant_05_Online_Autoencoder_QLearning
from _04_Visualisation.Visualisation import Visualisation
import logging


def main(main_path="", seed=0, chooseAutoencoder=2, epochs=8):

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

    # Convolutional Autoencoder Punishment Coefficients for Epochs: 2, 4, 6, 8, 20
    ae_model_2_pcs = {
        1:  [1.2, 1.5, 1.4, 0.9, 0.5],
        2:  [0.2, 0.6, 0.4, 0.4, 0.5],
        3:  [0.4, 0.5, 0.6, 0.5, 0.5],
        4:  [0.7, 1.3, 0.7, 1.1, 0.6],
        5:  [0.8, 1.4, 0.7, 1.2, 0.5],
        6:  [0.7, 0.7, 0.5, 0.4, 0.3],
        7:  [1.1, 0.8, 0.6, 0.5, 0.9],
        8:  [1.0, 0.9, 0.9, 0.7, 0.6],
        9:  [1.5, 1.0, 1.0, 1.5, 0.9],
        10: [1.3, 1.3, 1.2, 1.1, 0.7],
        11: [1.0, 1.3, 1.0, 1.1, 0.4],
        12: [0.9, 1.1, 1.0, 1.0, 0.4],
        13: [1.3, 1.4, 1.1, 1.1, 0.6],
        14: [1.3, 1.4, 1.1, 1.1, 0.8],
        15: [1.5, 1.4, 1.1, 1.2, 0.8],
        16: [1.5, 1.5, 1.5, 1.4, 1.4],
        17: [1.5, 1.4, 1.3, 1.5, 1.3],
        18: [1.5, 1.5, 1.5, 1.5, 1.2],
        19: [1.1, 1.0, 0.9, 0.4, 0.4],
        20: [1.0, 1.1, 1.0, 0.7, 0.5],
        21: [1.2, 1.0, 0.9, 1.0, 0.5],
        22: [1.4, 1.5, 1.3, 1.1, 0.9],
    }

    if epochs < 10:
        exp_path = f"{main_path}0{epochs}_Experiment_05_Epochs"
    else:
        exp_path = f"{main_path}{epochs}_Experiment_05_Epochs"

    variant_name = f"Variant_05_Epochs_{epochs}"

    if os.path.exists(exp_path) is False:
        os.mkdir(exp_path)

    for dataset in datasets:

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        path = datasets[dataset]

        if epochs == 2:
            punishment_coefficient = ae_model_2_pcs[dataset][0]
        elif epochs == 4:
            punishment_coefficient = ae_model_2_pcs[dataset][1]
        elif epochs == 6:
            punishment_coefficient = ae_model_2_pcs[dataset][2]
        elif epochs == 8:
            punishment_coefficient = ae_model_2_pcs[dataset][3]
        else:
            punishment_coefficient = ae_model_2_pcs[dataset][4]

        dataset_name = path[16:].split("/")
        variant_name = variant_name
        vis = Visualisation(variant_name, dataset_name, exp_path=f"{exp_path}/")
        vis_path = vis.get_visualisation_path()

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

        Variant_05_Online_Autoencoder_QLearning(path, vis, logger, parameter_logger,
                                                punishment_coefficient=punishment_coefficient,
                                                chooseAutoencoder=chooseAutoencoder,
                                                epochs=epochs
                                                )

        handler1.close()
        handler2.close()
        logger.removeHandler(handler1)
        parameter_logger.removeHandler(handler2)


if __name__ == '__main__':
    main()
