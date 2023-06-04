import os

import _Experiment_02
import _Experiment_04
import _Experiment_05
from multiprocessing import Process

from _04_Visualisation import Grid_Search_Table


def main():
    chooseAutoencoder = 2
    autoencoder_path = f"AE_Model_{chooseAutoencoder}"
    if os.path.exists(autoencoder_path) is False:
        os.mkdir(autoencoder_path)

    main_path = f"{autoencoder_path}/Epochs/"

    if os.path.exists(main_path) is False:
        os.mkdir(main_path)
    else:
        print(f"{main_path} already exists. Move, rename or remove it to run this experiment.")
        return

    variant_paths = []
    jobs = []
    epoch_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    variant_paths.append(f"{main_path}/V2/")
    variant_paths.append(f"{main_path}/V4/")
    variant_paths.append(f"{main_path}/V5/")

    for variant_path in variant_paths:
        if os.path.exists(variant_path) is False:
            os.mkdir(variant_path)

    for epochs in epoch_list:
        p2 = Process(target=_Experiment_02.main, args=(variant_paths[0], 0, chooseAutoencoder, epochs))
        p2.start()
        jobs.append(p2)

        p4 = Process(target=_Experiment_04.main, args=(variant_paths[1], 0, "", False, chooseAutoencoder, epochs))
        p4.start()
        jobs.append(p4)

        p5 = Process(target=_Experiment_05.main, args=(variant_paths[2], 0, "", False, False, False, False, False,
                                                      chooseAutoencoder, epochs))
        p5.start()
        jobs.append(p5)

        for job in jobs:
            job.join()

    for i in range(3):
        Grid_Search_Table.main(experiment_path=variant_paths[i], epoch_list=epoch_list)


if __name__ == '__main__':
    main()
