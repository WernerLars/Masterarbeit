import os

import Experiment_01
import Experiment_02
import Experiment_03
import Experiment_04
import Experiment_05
from multiprocessing import Process
from _04_Visualisation import Tables


def main():
    main_path = "Different_Variant_5/"

    if os.path.exists(main_path) is False:
        os.mkdir(main_path)
    else:
        return

    p1 = Process(target=Experiment_05.main, args=(main_path, 0, "", False, False, False))
    p1.start()
    p2 = Process(target=Experiment_05.main, args=(main_path, 0, "", True, False, False))
    p2.start()
    p3 = Process(target=Experiment_05.main, args=(main_path, 0, "", True, True, False))
    p3.start()
    p4 = Process(target=Experiment_05.main, args=(main_path, 0, "", True, True, True))
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()

    Tables.main(experiment_path=main_path)


if __name__ == '__main__':
    main()
