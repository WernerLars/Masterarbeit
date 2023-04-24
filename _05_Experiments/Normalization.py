import os

import Experiment_04
from multiprocessing import Process
from _04_Visualisation import Tables


def main():
    chooseAutoencoder = 1
    autoencoder_path = f"AE_Model_{chooseAutoencoder}"
    if os.path.exists(autoencoder_path) is False:
        os.mkdir(autoencoder_path)

    main_path = f"{autoencoder_path}/Normalization/"

    if os.path.exists(main_path) is False:
        os.mkdir(main_path)
    else:
        return

    p = Process(target=Experiment_04.main, args=(main_path, 0, "", True, chooseAutoencoder))
    p.start()
    p.join()

    Tables.main(experiment_path=main_path)


if __name__ == '__main__':
    main()
