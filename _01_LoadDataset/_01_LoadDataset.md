# _01_LoadDataset

LoadingDataset class has function loadData:
- uses extern code from spaike-project for loading Matlab files.
- does a max alignment
- returns spike class and y_labels

SpikeClassToPytorchDataset class converts spike class from spaike-project
into a dataset usable for Pytorch (oriented on the details from custom datasets in PyTorch: 
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)


