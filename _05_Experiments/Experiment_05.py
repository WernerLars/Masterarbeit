import os
import random

import numpy as np
import torch

from _03_SpikeSorter.Variant_05_Online_Autoencoder_QLearning import Variant_05_Online_Autoencoder_QLearning
from _04_Visualisation.Visualisation import Visualisation
import logging


def main():
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    datasets = {
        1: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise005.mat", 0.5],
        2: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise010.mat", 0.6],
        3: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise015.mat", 0.7],
        4: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise020.mat", 0.8],
        5: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise025.mat", 0.9],
        6: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise030.mat", 1],
        7: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise035.mat", 1.2],
        8: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise040.mat", 1.6],
        9: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy2_noise005.mat", 0.6],
        10: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy2_noise010.mat", 0.6],
        11: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy2_noise015.mat", 0.6],
        12: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy2_noise020.mat", 0.6],
        13: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult1_noise005.mat", 0.33],
        14: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult1_noise010.mat", 0.35],
        15: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult1_noise015.mat", 0.37],
        16: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult1_noise020.mat", 0.39],
        17: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult2_noise005.mat", 0.3],
        18: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult2_noise010.mat", 0.33],
        19: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult2_noise015.mat", 0.36],
        20: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult2_noise020.mat", 0.4],
        21: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Burst_Easy2_noise015.mat", 0.6],
        22: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Drift_Easy2_noise015.mat", 0.6],
    }

    variant_name = "Variant_05_Online_Autoencoder_QLearning"
    exp_path = "Experiment_05"
    if os.path.exists(exp_path) is False:
        os.mkdir(exp_path)

    normalise = False
    optimising = True
    templates = True
    noisy = False

    if normalise:
        variant_name = f"{variant_name}_norm"
    if optimising:
        variant_name = f"{variant_name}_opt"
    if templates:
        variant_name = f"{variant_name}_temp"
    if noisy:
        variant_name = f"{variant_name}_noisy"

    for dataset in datasets:
        print(variant_name)
        print(datasets[dataset])

        path = datasets[dataset][0]
        punishment_coefficient = datasets[dataset][1]
        dataset_name = path[16:].split("/")
        variant_name = variant_name
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
                                                normalise=normalise,
                                                templateMatching=templates,
                                                optimising=optimising,
                                                )

        handler1.close()
        handler2.close()
        logger.removeHandler(handler1)
        parameter_logger.removeHandler(handler2)


if __name__ == '__main__':
    main()
