import os

import _Experiment_05
from multiprocessing import Process
from _04_Visualisation import Tables
from tqdm import tqdm


def main():
    chooseAutoencoder = 2
    autoencoder_path = f"AE_Model_{chooseAutoencoder}"
    if os.path.exists(autoencoder_path) is False:
        os.mkdir(autoencoder_path)

    main_path = f"{autoencoder_path}/Random_Seeds_DV5/"

    if os.path.exists(main_path) is False:
        os.mkdir(main_path)
    else:
        print(f"{main_path} already exists. Move, rename or remove it to run this experiment.")
        return

    number_of_seeds = 10
    number_of_variants = 3
    variant_paths = []
    jobs = []

    for i in range(number_of_variants):
        variant_path = f"{main_path}/V5_{i+1}/"
        variant_paths.append(variant_path)
        if os.path.exists(variant_path) is False:
            os.mkdir(variant_path)

    with tqdm(range(number_of_seeds), desc="Random Seeds", position=0, leave=False) as rs_loop:
        for i in rs_loop:
            jobs = []
            p1 = Process(target=_Experiment_05.main,
                         args=(variant_paths[0], i, "", True, False, False, False, True, chooseAutoencoder,
                               8, 700, 1000, 1, True))
            p1.start()
            jobs.append(p1)
            p2 = Process(target=_Experiment_05.main,
                         args=(variant_paths[1], i, "", True, True, False, False, True, chooseAutoencoder,
                               8, 700, 1000, 2, True))
            p2.start()
            jobs.append(p2)
            p3 = Process(target=_Experiment_05.main,
                         args=(variant_paths[2], i, "", True, True, True, False, True, chooseAutoencoder,
                               8, 700, 1000, 3, True))
            p3.start()
            jobs.append(p3)

            for job in jobs:
                job.join()

    for i in range(number_of_variants):
        Tables.main(experiment_path=variant_paths[i], random_seeds=True)


if __name__ == '__main__':
    main()
