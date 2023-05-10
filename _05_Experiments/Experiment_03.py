import os
import random

import numpy as np
import torch

from _03_SpikeSorter.Variant_03_PCA_QLearning import Variant_03_PCA_QLearning
from _04_Visualisation.Visualisation import Visualisation
import logging


def main(main_path="", seed=0, pc=""):

    datasets = {
        1:  ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Burst_Easy2_noise015.mat", 1.0],
        2:  ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult1_noise005.mat",  0.4],
        3:  ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult1_noise010.mat",  0.6],
        4:  ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult1_noise015.mat",  1.0],
        5:  ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult1_noise020.mat",  1.5],
        6:  ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult2_noise005.mat",  0.4],
        7:  ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult2_noise010.mat",  0.8],
        8:  ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult2_noise015.mat",  1.0],
        9:  ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult2_noise020.mat",  1.2],
        10: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Drift_Easy2_noise015.mat", 1.2],
        11: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise005.mat",       1.2],
        12: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise010.mat",       1.3],
        13: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise015.mat",       1.4],
        14: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise020.mat",       1.4],
        15: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise025.mat",       1.4],
        16: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise030.mat",       1.5],
        17: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise035.mat",       1.5],
        18: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise040.mat",       1.5],
        19: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy2_noise005.mat",       0.7],
        20: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy2_noise010.mat",       0.8],
        21: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy2_noise015.mat",       1.1],
        22: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy2_noise020.mat",       1.4],
    }

    variant_name = "Variant_03_PCA_QLearning"

    if seed == 0:
        exp_path = f"{main_path}Experiment_03"
    else:
        exp_path = f"{main_path}Experiment_03_{seed}"

    if os.path.exists(exp_path) is False:
        os.mkdir(exp_path)

    for dataset in datasets:

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        path = datasets[dataset][0]

        if pc is not "":
            punishment_coefficient = pc
        else:
            punishment_coefficient = datasets[dataset][1]

        dataset_name = path[16:].split("/")
        vis = Visualisation(variant_name, dataset_name, exp_path=f"{exp_path}/", pc=f"{punishment_coefficient}")
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

        Variant_03_PCA_QLearning(path, vis, logger, parameter_logger,
                                 punishment_coefficient=punishment_coefficient,
                                 q_learning_size=300)

        handler1.close()
        handler2.close()
        logger.removeHandler(handler1)
        parameter_logger.removeHandler(handler2)


if __name__ == '__main__':
    main()
