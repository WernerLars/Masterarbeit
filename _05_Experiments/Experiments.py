from _03_SpikeSorter.Variant_01_PCA_KMeans import Variant_01_PCA_KMeans
from _03_SpikeSorter.Variant_02_Autoencoder_KMeans import Variant_02_Autoencoder_KMeans
from _03_SpikeSorter.Variant_03_PCA_QLearning import Variant_03_PCA_QLearning
from _03_SpikeSorter.Variant_04_Offline_Autoencoder_QLearning import Variant_04_Offline_Autoencoder_QLearning
from _03_SpikeSorter.Variant_05_Online_Autoencoder_QLearning import Variant_05_Online_Autoencoder_QLearning


def main():
    datasets = ["../_00_Datasets/01_SimDaten_Martinez2009/simulation_1.mat",
                "../_00_Datasets/03_SimDaten_Quiroga2020/004_C_Difficult1_noise005.mat",
                "../_00_Datasets/03_SimDaten_Quiroga2020/016_C_Easy1_noise005.mat"]

    path = datasets[2]
    variant = 5

    if variant == 1:
        Variant_01_PCA_KMeans(path)
    elif variant == 2:
        Variant_02_Autoencoder_KMeans(path)
    elif variant == 3:
        Variant_03_PCA_QLearning(path)
    elif variant == 4:
        Variant_04_Offline_Autoencoder_QLearning(path)
    elif variant == 5:
        Variant_05_Online_Autoencoder_QLearning(path)


if __name__ == '__main__':
    main()
