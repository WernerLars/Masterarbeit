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

    main_path = f"{autoencoder_path}/Grid_Search_PC/"

    if os.path.exists(main_path) is False:
        os.mkdir(main_path)
    else:
        print(f"{main_path} already exists. Move, rename or remove it to run this experiment.")
        return

    number_of_datasets = 22
    punishment_coefficients = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    list_of_variant_names = ["V3", "V4", "V5", "V5_1", "V5_2", "V5_3"]
    number_of_variants = len(list_of_variant_names)
    variant_paths = []

    for variant_name in list_of_variant_names:
        variant_path = f"{main_path}/{variant_name}/"
        variant_paths.append(variant_path)
        if os.path.exists(variant_path) is False:
            os.mkdir(variant_path)

    for d_index in tqdm(range(1, number_of_datasets + 1), desc="Datasets", position=0, leave=False):
        for v_index in tqdm(range(0, number_of_variants), desc="Variants", position=1, leave=False):
            jobs = []
            for pc in punishment_coefficients:
                p = Process(target=_Single_Experiment.main,
                            args=(variant_paths[v_index], d_index, v_index + 3, pc, True, chooseAutoencoder))
                p.start()
                jobs.append(p)
                time.sleep(30)

            for job in jobs:
                job.join()
                job.close()

    for i in range(number_of_variants):
        Grid_Search_Table.main(experiment_path=variant_paths[i])


if __name__ == '__main__':
    main()
