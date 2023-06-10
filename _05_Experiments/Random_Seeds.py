import os

import _Experiment_01
import _Experiment_02
import _Experiment_03
import _Experiment_04
import _Experiment_05
from multiprocessing import Process
from _04_Visualisation import Tables
from tqdm import tqdm


def main():
    chooseAutoencoder = 2
    autoencoder_path = f"AE_Model_{chooseAutoencoder}"
    if os.path.exists(autoencoder_path) is False:
        os.mkdir(autoencoder_path)

    main_path = f"{autoencoder_path}/Random_Seeds/"

    if os.path.exists(main_path) is False:
        os.mkdir(main_path)
    else:
        print(f"{main_path} already exists. Move, rename or remove it to run this experiment.")
        return

    number_of_seeds = 10
    number_of_variants = 5
    variant_paths = []

    for i in range(number_of_variants):
        variant_path = f"{main_path}/V{i + 1}/"
        variant_paths.append(variant_path)
        if os.path.exists(variant_path) is False:
            os.mkdir(variant_path)

    with tqdm(range(number_of_seeds), desc="Random Seeds", position=0, leave=False) as rs_loop:
        for i in rs_loop:
            jobs = []
            p1 = Process(target=_Experiment_01.main, args=(variant_paths[0], i, 1))
            p1.start()
            jobs.append(p1)
            p2 = Process(target=_Experiment_02.main, args=(variant_paths[1], i, chooseAutoencoder, 8, 2, True))
            p2.start()
            jobs.append(p2)
            p3 = Process(target=_Experiment_03.main, args=(variant_paths[2], i, "", 3, True))
            p3.start()
            jobs.append(p3)
            p4 = Process(target=_Experiment_04.main, args=(variant_paths[3], i, "", False, chooseAutoencoder, 8, 4, True))
            p4.start()
            jobs.append(p4)
            p5 = Process(target=_Experiment_05.main,
                         args=(variant_paths[4], i, "", False, False, False, False, False, chooseAutoencoder,
                               8, 700, 1000, 5, True))
            p5.start()
            jobs.append(p5)

            for job in jobs:
                job.join()

    for i in range(number_of_variants):
        Tables.main(experiment_path=variant_paths[i], random_seeds=True)


if __name__ == '__main__':
    main()
