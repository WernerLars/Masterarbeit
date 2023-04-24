import os

import Experiment_01
import Experiment_02
import Experiment_03
import Experiment_04
import Experiment_05
from multiprocessing import Process
from _04_Visualisation import Tables


def main():
    chooseAutoencoder = 1
    autoencoder_path = f"AE_Model_{chooseAutoencoder}"
    if os.path.exists(autoencoder_path) is False:
        os.mkdir(autoencoder_path)

    main_path = f"{autoencoder_path}/Base_Line/"

    if os.path.exists(main_path) is False:
        os.mkdir(main_path)
    else:
        return

    p1 = Process(target=Experiment_01.main, args=(main_path,))
    p1.start()
    p2 = Process(target=Experiment_02.main, args=(main_path, 0, chooseAutoencoder))
    p2.start()
    p3 = Process(target=Experiment_03.main, args=(main_path,))
    p3.start()
    p4 = Process(target=Experiment_04.main, args=(main_path, 0, "", False, chooseAutoencoder))
    p4.start()
    p5 = Process(target=Experiment_05.main,
                 args=(main_path, 0, "", False, False, False, False, False, chooseAutoencoder))
    p5.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()

    Tables.main(experiment_path=main_path)


if __name__ == '__main__':
    main()
