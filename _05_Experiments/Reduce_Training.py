import os

import Experiment_05
from multiprocessing import Process
from _04_Visualisation import Tables


def main():
    chooseAutoencoder = 2
    optimising = False
    autoencoder_path = f"AE_Model_{chooseAutoencoder}"
    if os.path.exists(autoencoder_path) is False:
        os.mkdir(autoencoder_path)

    if optimising:
        main_path = f"{autoencoder_path}/Reduce_Training_opt/"
    else:
        main_path = f"{autoencoder_path}/Reduce_Training/"

    if os.path.exists(main_path) is False:
        os.mkdir(main_path)
    else:
        return

    list_of_variant_names = ["V5_010", "V5_050", "V5_100", "V5_200", "V5_700"]
    number_of_variants = len(list_of_variant_names)
    variant_paths = []
    jobs = []

    for variant_name in list_of_variant_names:
        variant_path = f"{main_path}/{variant_name}/"
        variant_paths.append(variant_path)
        if os.path.exists(variant_path) is False:
            os.mkdir(variant_path)

    for i in range(0, number_of_variants):
        if i == 0:
            p1 = Process(target=Experiment_05.main,
                         args=(variant_paths[i], 0, "", optimising, False, False, False, False, chooseAutoencoder, 8, 10, 310))
            p1.start()
            jobs.append(p1)
        elif i == 1:
            p2 = Process(target=Experiment_05.main,
                         args=(variant_paths[i], 0, "", optimising, False, False, False, False, chooseAutoencoder, 8, 50, 350))
            p2.start()
            jobs.append(p2)
        elif i == 2:
            p3 = Process(target=Experiment_05.main,
                         args=(variant_paths[i], 0, "", optimising, False, False, False, False, chooseAutoencoder, 8, 100, 400))
            p3.start()
            jobs.append(p3)
        elif i == 3:
            p4 = Process(target=Experiment_05.main,
                         args=(variant_paths[i], 0, "", optimising, False, False, False, False, chooseAutoencoder, 8, 200, 500))
            p4.start()
            jobs.append(p4)
        else:
            p5 = Process(target=Experiment_05.main,
                         args=(variant_paths[i], 0, "", optimising, False, False, False, False, chooseAutoencoder, 8, 700, 1000))
            p5.start()
            jobs.append(p5)

    for job in jobs:
        job.join()

    Tables.main(experiment_path=main_path, random_seeds=False, minimal_distance_names=list_of_variant_names)


if __name__ == '__main__':
    main()
