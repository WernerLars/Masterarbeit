import os
import time

from _03_SpikeSorter.Variant_04_Offline_Autoencoder_QLearning import Variant_04_Offline_Autoencoder_QLearning
from _04_Visualisation.Visualisation import Visualisation
import logging


def main():
    datasets = {
        1: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise005.mat", 0.7],
        2: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise010.mat", 0.7],
        3: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise015.mat", 0.7],
        4: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise020.mat", 0.7],
        5: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise025.mat", 0.7],
        6: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise030.mat", 0.7],
        7: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise035.mat", 1],
        8: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy1_noise040.mat", 1.3],
        9: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy2_noise005.mat", 0.5],
        10: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy2_noise010.mat", 0.52],
        11: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy2_noise015.mat", 0.52],
        12: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Easy2_noise020.mat", 0.52],
        13: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult1_noise005.mat", 0.35],
        14: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult1_noise010.mat", 0.35],
        15: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult1_noise015.mat", 0.37],
        16: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult1_noise020.mat", 0.37],
        17: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult2_noise005.mat", 0.35],
        18: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult2_noise010.mat", 0.35],
        19: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult2_noise015.mat", 0.35],
        20: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Difficult2_noise020.mat", 0.35],
        21: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Burst_Easy2_noise015.mat", 0.54],
        22: ["../_00_Datasets/03_SimDaten_Quiroga2020/C_Drift_Easy2_noise015.mat", 0.54],
    }

    variant_name = "Variant_04_Offline_Autoencoder_QLearning"
    exp_name = "Experiment_04"
    print(exp_name)
    timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    exp_path = f"{exp_name}_{timestamp}"
    os.mkdir(exp_path)

    for dataset in datasets:

        print(variant_name)
        print(datasets[dataset])

        path = datasets[dataset][0]
        punishment_coefficient = datasets[dataset][1]
        dataset_name = path[16:].split("/")
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

        Variant_04_Offline_Autoencoder_QLearning(path, vis, logger, parameter_logger,
                                                 punishment_coefficient=punishment_coefficient)

        handler1.close()
        handler2.close()
        logger.removeHandler(handler1)
        parameter_logger.removeHandler(handler2)


if __name__ == '__main__':
    main()
