# _02_Classes_Autoencoder_QLearning

Implementation of Autoencoder Models, Dyna-Q-Learning and Templates.

### Autoencoder
- classical and convolutional autoencoder architectures
- uses PyTorch and nn Module
- both approach have four layers (two in encoder and decoder)
  - classical uses Linear Layers and input size gets divided by 4 
  - convolutional uses Conv1d and Conv1dTranspose Layers 
  - both use Leaky Relu as activation function
- convolutional autoencoder needs reshape before input in encoder and between encoder/decoder
  - computation of dimension with formula: (Input Size - Kernel Size + 2*P)/S  +  1
- encoded (features) and decoded (reconstructed spikes) are send back 
- print Model method makes graphs and ONNX Files for both architectures
  - graphs are made with torchview (draw_graph method + render)
  - graphs and ONNX Files are saved in Architecture_Files folder
  - ONNX Files can be opened with Netron (https://github.com/lutzroeder/netron)

### QLearning




### Templates
- list for saving mean waveforms of clusters
- appending spike if new cluster is added
- otherwise computes mean template for every dimension of incoming spike
- mean is computed new-old with a learning rate to limit change of mean template

