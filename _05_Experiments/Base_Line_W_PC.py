import os

import Experiment_01
import Experiment_02
import Experiment_03
import Experiment_04
import Experiment_05
from multiprocessing import Process
from _04_Visualisation import Tables


def main():
    main_path = f"Base_Line_W_PC/"

    if os.path.exists(main_path) is False:
        os.mkdir(main_path)
    else:
        return

    punishment_coefficients = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    variant_paths = []
    jobs = []

    for pc in punishment_coefficients:
        variant_path = f"{main_path}/{pc}/"
        variant_paths.append(variant_path)
        if os.path.exists(variant_path) is False:
            os.mkdir(variant_path)

        p1 = Process(target=Experiment_01.main, args=(variant_path,))
        p1.start()
        jobs.append(p1)
        p2 = Process(target=Experiment_02.main, args=(variant_path,))
        p2.start()
        jobs.append(p2)
        p3 = Process(target=Experiment_03.main, args=(variant_path, 0, pc))
        p3.start()
        jobs.append(p3)
        p4 = Process(target=Experiment_04.main, args=(variant_path, 0, pc))
        p4.start()
        jobs.append(p4)
        p5 = Process(target=Experiment_05.main, args=(variant_path, 0, pc))
        p5.start()
        jobs.append(p5)

        for job in jobs:
            job.join()

    for variant_path in variant_paths:
        Tables.main(experiment_path=variant_path)


if __name__ == '__main__':
    main()
