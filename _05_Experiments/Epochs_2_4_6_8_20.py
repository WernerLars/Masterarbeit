import os

from multiprocessing import Process

from _04_Visualisation import Tables
import _Experiment_05_Epochs


def main():
    chooseAutoencoder = 2
    autoencoder_path = f"AE_Model_{chooseAutoencoder}"
    if os.path.exists(autoencoder_path) is False:
        os.mkdir(autoencoder_path)

    main_path = f"{autoencoder_path}/Epochs_2_4_6_8_20/"

    if os.path.exists(main_path) is False:
        os.mkdir(main_path)
    else:
        print(f"{main_path} already exists. Move, rename or remove it to run this experiment.")
        return

    jobs = []
    epoch_list = [2, 4, 6, 8, 20]
    position = 0

    for epochs in epoch_list:
        p = Process(target=_Experiment_05_Epochs.main, args=(main_path, 0, chooseAutoencoder, epochs, position, True))
        p.start()
        jobs.append(p)
        position += 1

    for job in jobs:
        job.join()

    Tables.main(experiment_path=main_path)


if __name__ == '__main__':
    main()
