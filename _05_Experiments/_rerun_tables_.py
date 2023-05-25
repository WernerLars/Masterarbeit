import os

from _04_Visualisation import Tables, Grid_Search_Table


def main():
    for chooseAutoencoder in [1, 2]:
        autoencoder_path = f"AE_Model_{chooseAutoencoder}"

        if os.path.exists(f"{autoencoder_path}/Normalization"):
            Tables.main(experiment_path=f"{autoencoder_path}/Normalization")

        if os.path.exists(f"{autoencoder_path}/Base_Line"):
            Tables.main(experiment_path=f"{autoencoder_path}/Base_Line")

        list_of_variant_names = ["V3", "V4", "V5", "V5_1", "V5_2", "V5_3"]
        for variant_name in list_of_variant_names:
            main_path = f"{autoencoder_path}/Grid_Search_PC/{variant_name}"
            if os.path.exists(main_path):
                Grid_Search_Table.main(experiment_path=main_path)

        for i in range(5):
            main_path = f"{autoencoder_path}/Random_Seeds/V{i + 1}"
            if os.path.exists(main_path):
                Tables.main(experiment_path=main_path, random_seeds=True)

        if os.path.exists(f"{autoencoder_path}/Different_Variant_5"):
            Tables.main(experiment_path=f"{autoencoder_path}/Different_Variant_5")

        for i in range(3):
            main_path = f"{autoencoder_path}/Random_Seeds_DV5/V5_{i + 1}"
            if os.path.exists(main_path):
                Tables.main(experiment_path=main_path, random_seeds=True)

        if os.path.exists(f"{autoencoder_path}/Epochs"):
            epoch_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
            Grid_Search_Table.main(experiment_path=f"{autoencoder_path}/Epochs/V2", epoch_list=epoch_list)
            Grid_Search_Table.main(experiment_path=f"{autoencoder_path}/Epochs/V4", epoch_list=epoch_list)
            Grid_Search_Table.main(experiment_path=f"{autoencoder_path}/Epochs/V5", epoch_list=epoch_list)

        if os.path.exists(f"{autoencoder_path}/Epochs_GS_PC"):
            Grid_Search_Table.main(experiment_path=f"{autoencoder_path}/Epochs_GS_PC/V5_2")
            Grid_Search_Table.main(experiment_path=f"{autoencoder_path}/Epochs_GS_PC/V5_4")
            Grid_Search_Table.main(experiment_path=f"{autoencoder_path}/Epochs_GS_PC/V5_6")
            Grid_Search_Table.main(experiment_path=f"{autoencoder_path}/Epochs_GS_PC/V5_20")

        if os.path.exists(f"{autoencoder_path}/Epochs_2_4_6_8_20"):
            Tables.main(experiment_path=f"{autoencoder_path}/Epochs_2_4_6_8_20")

        list_of_variant_names = ["V5_010", "V5_050", "V5_100", "V5_200"]
        if os.path.exists(f"{autoencoder_path}/Reduce_Training"):
            Tables.main(experiment_path=f"{autoencoder_path}/Reduce_Training", random_seeds=False,
                        minimal_distance_names=list_of_variant_names)

        if os.path.exists(f"{autoencoder_path}/Reduce_Training_opt"):
            Tables.main(experiment_path=f"{autoencoder_path}/Reduce_Training_opt", random_seeds=False,
                        minimal_distance_names=list_of_variant_names)


if __name__ == '__main__':
    main()
