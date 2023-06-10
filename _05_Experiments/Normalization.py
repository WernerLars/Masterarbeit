import os

import _Experiment_04
from multiprocessing import Process
from _04_Visualisation import Tables


def main():
    chooseAutoencoder = 2
    autoencoder_path = f"AE_Model_{chooseAutoencoder}"
    if os.path.exists(autoencoder_path) is False:
        os.mkdir(autoencoder_path)

    main_path = f"{autoencoder_path}/Normalization/"

    if os.path.exists(main_path) is False:
        os.mkdir(main_path)
    else:
        print(f"{main_path} already exists. Move, rename or remove it to run this experiment.")
        return

    p1 = Process(target=_Experiment_04.main, args=(main_path, 0, "", False, chooseAutoencoder, 8, 0, True))
    p1.start()

    p2 = Process(target=_Experiment_04.main, args=(main_path, 0, "", True, chooseAutoencoder, 8, 1, True))
    p2.start()

    p1.join()
    p2.join()

    Tables.main(experiment_path=main_path)


if __name__ == '__main__':
    main()
