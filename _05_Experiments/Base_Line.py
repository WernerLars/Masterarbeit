import os

import Experiment_01
import Experiment_02
import Experiment_03
import Experiment_04
import Experiment_05
from multiprocessing import Process
from _04_Visualisation import Tables


def main():
    main_path = "Base_Line/"

    if os.path.exists(main_path) is False:
        os.mkdir(main_path)
    else:
        return

    p1 = Process(target=Experiment_01.main, args=(main_path,))
    p1.start()
    p2 = Process(target=Experiment_02.main, args=(main_path,))
    p2.start()
    p3 = Process(target=Experiment_03.main, args=(main_path,))
    p3.start()
    p4 = Process(target=Experiment_04.main, args=(main_path,))
    p4.start()
    p5 = Process(target=Experiment_05.main, args=(main_path,))
    p5.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()

    Tables.main(experiment_path=main_path)


if __name__ == '__main__':
    main()
