# _06_Tests

Here Test cases for loading a dataset, converting it into a pytorch dataset and workflow of q-learning are stored.

### Test_LoadingDataset
- here the LoadingDataset class is tested
- setup initialises this class with path of dataset and logger
- calls load_data method
- asserts if specific types of dataloader attributes are correctly set


### Test_QLearning
- here the QLearning class is tested
- setup initialises this class with logger and the number of features
- goal is to test the functionality of methods of q-learning
  - if values in q-table and model are set correctly
  - if reset of q-table and model works
  - if a new cluster is added to q-table and model assert if new key and new column is added to both
  - if episode number and rewards are computed correctly

### Test_SpikeClassToPytorchDataset
- here SpikeClassToPytorchDataset is tested
- setup initialises LoadDataset and output is send to init method
- init setups SpikeClassToPytorchDataset class with aligned_spikes and y_labels
- it checks if SpikeClass dataset is correctly converted into Pytorch dataset
  - asserts if attributes are correctly transferred (y_labels -> cluster; aligned_spikes -> data)
  - asserts length of y_labels