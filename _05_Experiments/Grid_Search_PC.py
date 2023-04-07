import os
import time

from tqdm import tqdm
import numpy as np

import Single_Experiment
from multiprocessing import Process
from _04_Visualisation import Grid_Search_Table


def main():
    main_path = "Grid_Search_PC/"

    if os.path.exists(main_path) is False:
        os.mkdir(main_path)
    else:
        return

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
            for pc in np.arange(0.2, 1.401, 0.1):
                p = Process(target=Single_Experiment.main, args=(variant_paths[v_index], d_index, v_index+3, pc, True))
                p.start()
                jobs.append(p)
                time.sleep(3)

            for job in jobs:
                job.join()

    for i in range(number_of_variants):
        Grid_Search_Table.main(experiment_path=variant_paths[i])


if __name__ == '__main__':
    main()
