import os
import time

from tqdm import tqdm

import _Single_Experiment
from multiprocessing import Process
from _04_Visualisation import Grid_Search_Table


def main():
    chooseAutoencoder = 2
    autoencoder_path = f"AE_Model_{chooseAutoencoder}"
    if os.path.exists(autoencoder_path) is False:
        os.mkdir(autoencoder_path)

    main_path = f"{autoencoder_path}/Epochs_GS_PC/"

    if os.path.exists(main_path) is False:
        os.mkdir(main_path)
    else:
        print(f"{main_path} already exists. Move, rename or remove it to run this experiment.")
        return

    number_of_datasets = 22
    punishment_coefficients = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    list_of_variant_names = ["V5_2", "V5_4", "V5_6", "V5_20"]
    epochs = [2, 4, 6, 20]
    variant_paths = []

    for variant_name in list_of_variant_names:
        variant_path = f"{main_path}/{variant_name}/"
        variant_paths.append(variant_path)
        if os.path.exists(variant_path) is False:
            os.mkdir(variant_path)

    for d_index in tqdm(range(1, number_of_datasets + 1), position=0, desc="Datasets"):
        for v_index in tqdm(range(0, len(list_of_variant_names)), position=1, desc="Variants"):
            jobs = []
            for pc in punishment_coefficients:
                p = Process(target=_Single_Experiment.main,
                            args=(variant_paths[v_index], d_index, 5, pc, True, chooseAutoencoder, epochs[v_index]))
                p.start()
                jobs.append(p)
                time.sleep(10)

            for job in jobs:
                job.join()

    for i in range(len(list_of_variant_names)):
        Grid_Search_Table.main(experiment_path=variant_paths[i])


if __name__ == '__main__':
    main()
