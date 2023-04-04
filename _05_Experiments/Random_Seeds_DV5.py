import os

import Experiment_01
import Experiment_02
import Experiment_03
import Experiment_04
import Experiment_05
from multiprocessing import Process
from _04_Visualisation import Tables


def main():
    main_path = "Random_Seeds_DV5/"

    if os.path.exists(main_path) is False:
        os.mkdir(main_path)
    else:
        return

    number_of_seeds = 9
    number_of_variants = 3
    variant_paths = []
    jobs = []

    for i in range(number_of_variants):
        variant_path = f"{main_path}/V5_{i+1}/"
        variant_paths.append(variant_path)
        if os.path.exists(variant_path) is False:
            os.mkdir(variant_path)

    for i in range(number_of_seeds):

        p1 = Process(target=Experiment_05.main, args=(variant_paths[0], i, "", True, False, False))
        p1.start()
        jobs.append(p1)
        p2 = Process(target=Experiment_05.main, args=(variant_paths[1], i, "", True, True, False))
        p2.start()
        jobs.append(p2)
        p3 = Process(target=Experiment_05.main, args=(variant_paths[2], i, "", True, True, True))
        p3.start()
        jobs.append(p3)

    for job in jobs:
        job.join()

    for i in range(number_of_variants):
        Tables.main(experiment_path=variant_paths[i])


if __name__ == '__main__':
    main()
