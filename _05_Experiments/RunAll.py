import Experiment_01
import Experiment_02
import Experiment_03
import Experiment_04
import Experiment_05
from multiprocessing import Process


def main():
    p1 = Process(target=Experiment_01.main())
    p1.start()
    p2 = Process(target=Experiment_02.main())
    p2.start()
    p3 = Process(target=Experiment_03.main())
    p3.start()
    p4 = Process(target=Experiment_04.main())
    p4.start()
    p5 = Process(target=Experiment_05.main())
    p5.start()


if __name__ == '__main__':
    main()
