# _03_SpikeSorter

Here you can find the implementation of the five (+ three) Spike Sorter Variants. They are combinations 
of two feature extraction methods with PCA and Autoencoder with the two clustering method K-Means and Dyna-Q-Learning.
Variant 4 and 5 use different offline or online training strategy for autoencoder.
Every class have an initialisation part where hyperparameters are set and dataset is loaded. 
The extracted features from feature extraction method are passed to clustering method. 
The resulting cluster labels are passed to visualisation class for creating contingency matrix and computing accuracy.

### 1. PCA with KMeans
- uses sklearn implementations of PCA and KMeans
- some additional hyperparameters
  - pca_components are the number of principal components which are extracted from PCA
  - number_of_clusters is a given parameter for K-Means extracted from true labels
### 2. Autoencoder with KMeans
- in all Autoencoder Variants the Hyperparameter chooseAutoencoder decides which autoencoder architecture is used
  - 1: classical 2: convolutional
- computation of input size for first layer of autoencoder from length of a sample
- number_of_features constrains output layer of encoder and input layer of decoder
  - default value is 2
- loss function in all variants is Mean Squared Error 
- optimizer in all variants is Adams
- uses offline training for autoencoder
  - epochs (default value is 8) is the number of training periods over training dataset
  - batch size (default value is 1) defines how many spikes are used before updating weights
  - in every epoch all spikes of training dataset are used as batches to train autoencoder 
  - loss is computed with comparing the difference between original and reconstructed spike
  - with computed loss backpropagation is carried out
- loss graphs are printed for every epoch and over all epochs
  - for every epoch the loss from every 100 spikes will be used
  - for all epochs the mean loss over that epoch is used
- in clustering torch.no_grad needs to be used to convert tensors back to numpy arrays
### 3. PCA with Dyna-Q-Learning
- uses PCA from sklearn and QLearning class
- pca_components are the number of principal components which are extracted from PCA
- q-learning size defines the number of spikes used in clustering (default is 300)
- if normalization is used the method normaliseFeatures from QLearning class is called
  - therefore it is necessary to feed two spikes into feature list of QLearning by using addToFeatureSet method
  - if feature list would be empty or contains one spike, max-min value is 0 which would lead to divide by zero exception
### 4. Offline Autoencoder with Dyna-Q-Learning
- combines autoencoder with dyna-q-learning 
- split dataset into training and test data by split ratio of 0.9
- offline training like in variant 2
- only in this variant it is possible to visualize normalization feature spaces
### 5. Online Autoencoder with Dyna-Q-Learning
- no split of dataset
- uses two training phases which are constrained by maxAutoencoderTraining and maxTraining
  - maxAutoencoderTraining defines the number of spikes used to only train the autoencoder (first training phase)
    - default value is 700
  - maxTraining defines the maximal number of spikes used from dataset
    - similar to q-learning size 
    - default value is 1000
  - maxTraining - maxAutoencoderTraining gives the number of spikes used in clustering (second training phase) 
    - this value is 1000-700=300
- online training
  - we don't know the complete dataset 
  - autoencoder is trained on incoming batch/spike
    - epochs define how often autoencoder is trained on that specific batch/spike
    - after training the batch is dropped and not trained again
- in second training phase it is default (normal) to only do clustering
  - optimising approaches expand second training phase

#### Optimising approaches:
- optimising: trains also in second training phase
  - uses optimisingAutoencoder to train while encoder is used for clustering
  - weights of optimisingAutoencoder are saved and loaded to encoder with updateFactor
    - default value is 1, so it updates the weights after every spike
- template matching: computes mean templates which are used to train optimisingAutoencoder
  - incoming spike updates mean template of the cluster
  - batch for training contains mean templates of clusters with exception that incoming spike replaces the template of its cluster
- noisy: noiseFactor gets added to all mean templates
  - default value is 0.001 
  - interval ranges from -noiseFactor to +noiseFactor in uniform distribution
- template matching works only if optimising is used
- noisy works only if optimising and template matching is used