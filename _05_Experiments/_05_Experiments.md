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
- calls _Experiment_04 with and without normalization
#### Base_Line_W_PC
- uses a fixed punishment coefficient to call every variant dataset combination
- Experiments are called with a fixed pc in range from 0.1 to 1.5 with 0.1 steps
- every punishment coefficient gets a separate folder
- Tables class is called to show accuracy results
#### Grid_Search_PC
- search for individual punishment coefficient for every variant dataset combination
- Grid Search is performed on a range from 0.1 to 1.5 with 0.1 steps
- Performed on _Experiment03-_Experiment05 and for three optimising approaches (different variant 5) from Q3
  - they are performed on every variant independently and saved in separate folders V3-V5_3
- Grid Search Table class is called
- Best punishment coefficients are saved for every variant in Best_Punishment_Coefficients.txt
#### Base_Line
- calling _Experiment_01-_Experiment_05 with individually found punishment coefficients of Grid_Search_PC
  - results of Grid_Search_PC are manually written into dataset dictionary in Experiment03-Experiment05 
  - second column are pc for classical autoencoder and third column are pc for convolutional autoencoder
  - Experiment03 only has two columns, because it uses PCA as feature extraction
___

### Q2: How stable do the variants run on differently chosen random seeds?
#### Random_Seeds
- testing 10 Random Seeds ranging from 0 to 9 in 1 steps
- calling Experiment_01-Experiment_05 with fixed random seed and individual punishment coefficients
- every variant gets a folder for results
- Tables class is called for computing results
- PCA-KMeans is stable
- classical autoencoder is unstable (above all seed 6) in comparison to convolutional autoencoder
  - in the experiments from Q3 and Q4 the classical autoencoder is dropped
- Dyna-Q-Learning is slightly sensitive to Random Seeds (see V3)
  - Random Seed 0, which has optimised punishment coefficients from Grid Search, performs slightly better than all other random seeds.
- Tables class is called with boolean random_seeds as True
  - changes x-Axis names of accuracy table to random seed
___

### Q3: Do the proposed optimisation approaches improve the online variant?
#### Different_Variant_5
- four different variant 5 
  - normal: standard Variant 5 (no changes)
  - optimising: in Clustering a second training phase takes place, where the clustered spikes are also used to train an optimising model
    - optimising model updates encoder with an updateFactor 
    - default value for updateFactor is 1
  - template matching: mean template for every cluster is created and used to train the autoencoder
    - only in second training phase
    - every new spike updates template
    - all mean templates except the cluster with new incoming spike are used as batch (new spike replaces template in batch)
  - noisy batches: adding noise to mean templates in batch
    - noise is added as a noise Factor ranging from -noiseFactor to +noiseFactor in uniform distribution 
    - default values for noise Factor is 1

- Grid_Search_PC also searches for individual punishment coefficients for the different variant 5 
  - are manually added to Experiment_05 as ae_model_2 dictionary object
  - only available for convolutional autoencoder
  - classical autoencoder uses pc from normal variant 5
- the four variants are called with boolean values in Experiment05
  - first three boolean values refer to optimising, templates, noisy
    - templates work only if optimising is True
    - noisy works only if optimising and templates are True
  - last two refer to normalise and random seed
- Different_Variant_5 experiment uses Experiment05 to call the four different variant 5
- Tables class is called for results
#### Random_Seeds_DV5
- like in Q2 Random Seeds are tested for the four different variant 5
- testing 10 Random Seeds ranging from 0 to 9 in 1 steps
- calling Experiment05 with fixed random seed and individual punishment coefficients
  - here also the pc from ae_model_2 dictionary are used
- Tables class is called with boolean random_seeds as True
  - changes x-Axis names of accuracy table to random seed
___

### Q4: Which hyperparameters of the online variant can be reduced to still achieve good results?
#### Epochs
- testing autoencoder on different epoch numbers (2,4,6,8,10,12,14,16,18,20)
- only Experiment_02, Experiment_04 and Experiment_05 are called
  - testing only normal variant 05
- Grid Search Table class is called with above-mentioned epochs for x-axis names
- not optimised punishment coefficients so no exact results 
#### Epochs_GS_PC
- specialising only on epochs 2,4,6,8 and 20 for a Grid Search
- only on _Experiment_05
- called with a fixed pc in range from 0.1 to 1.5 with 0.1 steps
- resulting optimal pc are extracted in written into dictionary in Experiment_05_Epochs
  - also only available for convolutional autoencoder
  - _Experiment_05_Epochs works like _Experiment_05
  - used for Epochs_2_4_6_8_20
#### Epochs_2_4_6_8_20
- calling _Experiment_05_Epochs with epochs 2,4,6,8 and 20
- found punishment coefficients are used for specific epoch
- Tables class for results
#### Reduce_Training
- testing if hyperparameter maxAutoencoderTraining can be reduced
- normal and optimising variant 5 are tested
- maxAutoencoderTraining values of 10,50,100 and 200 are tested
  - maxTraining values are 300 apart for clustering, so they are adjusted to 310,350,400 and 500.
- introducing minimal cluster distance as metric for comparing feature spaces
  - comments on computing the metric are found in Visualisation class
- separated results for normal and optimising variant
  - boolean optimising needs to be set in Reduce_Training
  - results as table in Minimal_Cluster_Distances.txt file

