import os

from _04_Visualisation import Tables, Grid_Search_Table


def main():
    for chooseAutoencoder in [1, 2]:
        autoencoder_path = f"AE_Model_{chooseAutoencoder}"

        if os.path.exists(f"{autoencoder_path}/Normalization"):
            Tables.main(experiment_path=f"{autoencoder_path}/Normalization")

        if os.path.exists(f"{autoencoder_path}/Base_Line"):
            Tables.main(experiment_path=f"{autoencoder_path}/Base_Line")

        punishment_coefficients = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        for pc in punishment_coefficients:
            main_path = f"{autoencoder_path}/Base_Line_W_PC/{pc}"
            if os.path.exists(main_path):
                Tables.main(experiment_path=main_path)

        for i in range(3):
            main_path = f"{autoencoder_path}/Grid_Search_PC/V{i + 3}"
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


if __name__ == '__main__':
    main()
