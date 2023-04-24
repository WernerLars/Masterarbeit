import os
import time

from tqdm import tqdm

import Single_Experiment
from multiprocessing import Process
from _04_Visualisation import Grid_Search_Table


def main():
    chooseAutoencoder = 1
    autoencoder_path = f"AE_Model_{chooseAutoencoder}"
    if os.path.exists(autoencoder_path) is False:
        os.mkdir(autoencoder_path)

    main_path = f"{autoencoder_path}/Grid_Search_PC/"

    if os.path.exists(main_path) is False:
        os.mkdir(main_path)
    else:
        return

    punishment_coefficients = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    number_of_variants = 3
    number_of_datasets = 22
    variant_paths = []

    for i in range(number_of_variants):
        variant_path = f"{main_path}/V{i + 3}/"
        variant_paths.append(variant_path)
        if os.path.exists(variant_path) is False:
            os.mkdir(variant_path)

    for d_index in tqdm(range(1, number_of_datasets + 1), position=0, desc="Datasets"):
        for v_index in tqdm(range(0, number_of_variants), position=1, desc="Variants"):
            jobs = []
            for pc in punishment_coefficients:
                p = Process(target=Single_Experiment.main,
                            args=(variant_paths[v_index], d_index, v_index+3, pc, True, chooseAutoencoder))
                p.start()
                jobs.append(p)
                time.sleep(3)

            for job in jobs:
                job.join()

    for i in range(number_of_variants):
        Grid_Search_Table.main(experiment_path=variant_paths[i])


if __name__ == '__main__':
    main()
