# MA Lars Werner Code

Implementation of Spike Sorter Pipelines consisting Autoencoder as Feature Extraction and 
Dyna-Q-Learning as Clustering Method. There are multiple variant combinations with PCA and 
K-Means for comparing the results:
- Variant 01: PCA + KMeans
- Variant 02: Autoencoder + KMeans
- Variant 03: PCA + Dyna-Q-Learning
- Variant 04: Offline Autoencoder + Dyna-Q-Learning
- Variant 05: Online Autoencoder + Dyna-Q-Learning

## Installation Details

This project is build on Windows 11 with Python 3.7, so it is recommended 
to use these versions to reduce compatibility problems.

For installation of necessary packages a requirements.txt file is provided.

In Experiments are long path created, so it is recommended to disable path limit 
(in Windows it can be done in installation from python or in registry).

## Project Structure

Every Folder in this project has a Markdown File explaining in detail what the files are used for.

- _00_Datasets: Here the datasets are stored (Quiroga 2020)
- _01_LoadDataset: External files to load data from mat files and to convert data to Pytorch Dataloader
- _02_Classes_Autoencoder_QLearning: Autoencoder models, QLearning and templates classes, which are used in spike sorters
- _03_SpikeSorter: five variants of spike sorter pipelines, which are called in experiments
- _04_Visualisation: printing graphs and tables for visualising results of spike sorters and experiments
- _05_Experiments: run files of this project, consists experiments for specific research questions
- _06_Tests: some tests for checking if loading data and q learning works as intended

## License

This project has a MIT License. Details of permission and usage can be seen in LICENSE.txt File.

