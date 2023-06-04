# _05_Experiments

In this folder the experiments for this project are executed and the results are stored in specified folders.
- in AE_Model_1 results of classical autoencoder are stored
  - in this case chooseAutoencoder has to be set as 1
- in AE_Model_2 results of convolutional autoencoder are stored
  - in this case chooseAutoencoder has to be set as 2

Important details:
- check if chooseAutoencoder is set correctly before executing an experiment
  tqdm only works correctly, if Experiment01-Experiment05 are called separately (no multiprocessing)
  - for most of the experiments multiprocessing is used, so tqdm is disabled
  - otherwise process bars of tqdm will overlap
- only run one experiment at once because of multiprocessing and hardware limits especially if Grid Search experiments are executed. 
- Removing low punishment coefficients like 0.1 and 0.2 will greatly improve computation times in Grid Search experiments.
- _Experiment_05 stores punishment coefficients for Different_Variant_5 (normal,optimising,temp,noisy) only for convolutional autoencoder
  - if classical autoencoder is used for different variant 5 experiment the ones from baseline are used (no grid search was performed)
- _Experiment_05_Epochs stores punishment coefficients for Epochs_2_4_6_8_20 on normal variant 5 only for convolutional autoencoder.
  - classical autoencoder is not supported for Epochs_2_4_6_8_20 experiment (would use punishment coefficients of convolutional autoencoder).
- minimal cluster distances are only visible for Reduce_Training experiment
  - if this metric is needed in other experiments, they need to be rerun so that the minimal cluster distance gets written into log files.
- rerun_tables method is to rerun Grid_Search_Table and Tables classes to update results for all experiments
- _Experiment_01 - _Experiment_05 perform spike sorting with the specific variant and all datasets; it initializes visualisation and logs
- _Experiment_05_Epochs is like _Experiment_05 and only used for Epochs_2_4_6_8_20 experiment (extra saved punishment coefficients)


In the following the research questions of the master thesis are stated and the corresponding experiments are shortly described. 
For more details and discussion about the experiments look into chapter 4 and 5 of the master thesis.
___

### Q1: How do different variants of autoencoder and Q-learning perform compared to PCA with K-Means?
#### Normalization 
- to answer the question if a normalization should be used for autoencoder features
- visualisation of normalization feature space only available in variant 04
- calls Experiment04 with and without normalization
#### Base_Line_W_PC
- uses a fixed punishment coefficient to call every variant dataset combination
- Experiments are called with a fixed pc in range from 0.1 to 1.5 with 0.1 steps
- every punishment coefficient gets a separate folder
- Tables class is called to show accuracy results
#### Grid_Search_PC
- search for individual punishment coefficient for every variant dataset combination
- Grid Search is performed on a range from 0.1 to 1.5 with 0.1 steps
- Performed on Experiment03-Experiment05 and for three optimising approaches (different variant 5) from Q3
  - they are performed on every variant independently and saved in separate folders V3-V5_3
- Grid Search Table class is called
- Best punishment coefficients are saved for every variant in Best_Punishment_Coefficients.txt
#### Base_Line
- calling Experiment01-Experiment05 with individually found punishment coefficients of Grid_Search_PC
  - results of Grid_Search_PC are manually written into dataset dictionary in Experiment03-Experiment05 
  - second column are pc for classical autoencoder and third column are pc for convolutional autoencoder
  - Experiment03 only has two columns, because it uses PCA as feature extraction
___

### Q2: How stable do the variants run on differently chosen random seeds?
#### Random_Seeds


___

### Q3: Do the proposed optimisation approaches improve the online variant?
#### Different_Variant_5
#### Random_Seeds_DV5

___

### Q4: Which hyperparameters of the online variant can be reduced to still achieve good results?
#### Epochs
#### Epochs_GS_PC
#### Epochs_2_4_6_8_20
#### Reduce_Training


