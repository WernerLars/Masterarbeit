import os

import _Experiment_05
from multiprocessing import Process
from _04_Visualisation import Tables


def main():
    chooseAutoencoder = 2
    autoencoder_path = f"AE_Model_{chooseAutoencoder}"
    if os.path.exists(autoencoder_path) is False:
        os.mkdir(autoencoder_path)

    main_path = f"{autoencoder_path}/Different_Variant_5/"

    if os.path.exists(main_path) is False:
        os.mkdir(main_path)
    else:
        print(f"{main_path} already exists. Move, rename or remove it to run this experiment.")
        return

    p1 = Process(target=_Experiment_05.main,
                 args=(main_path, 0, "", False, False, False, False, False, chooseAutoencoder))
    p1.start()
    p2 = Process(target=_Experiment_05.main,
                 args=(main_path, 0, "", True, False, False, False, False, chooseAutoencoder))
    p2.start()
    p3 = Process(target=_Experiment_05.main,
                 args=(main_path, 0, "", True, True, False, False, False, chooseAutoencoder))
    p3.start()
    p4 = Process(target=_Experiment_05.main,
                 args=(main_path, 0, "", True, True, True, False, False, chooseAutoencoder))
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()

    Tables.main(experiment_path=main_path)


if __name__ == '__main__':
    main()
