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
- performs Dyna-Q-Learning as a clustering method
- Q-Table and Model are dictionaries
  - Q-Table states are new_cluster,c1,c2.. and actions are 0,1,2.. (action 0 refers to new_cluster)
  - keys of the dictionaries are the states
  - in Q-Table the reward is saved as a list, which can be extracted with the action
  - in Model reward and next state are saved in a list of lists
  - for every new cluster a new dictionary entry for Q-Table and Model are appended
    - to that a new action element is added in the lists 
- randomFeatures is also a dictionary which contains features of clusters
  - for every cluster a list with lists is created (every list is a dimension)
  - maxRandomFeatures constrains the number of elements in a dimension list (default is 60)
    - every additional feature, the first added element is deleted and the new one is appended
- reset q-learning will reset q-table and model to its origin form
  - states and actions in q-table and model are lost
- reset q-table or model will only reset values like rewards and new state
  - states and actions in q-table and model will remain
- print q-table and model functions creates dataframe and prints out the table in log
- epsilon-greedy method decides if the agent explores or exploits
  - exploring means a random action is taken, exploiting means the max action is taken
- computation of reward depends on action (rewards in this case are punishments)
  - creating new cluster (0): uses punishment coefficient (pc) to compute reward
    - low pc leads to creating many clusters
    - high pc leads to creating only one or two clusters
  - otherwise the action is used to extract randomFeatures from dictionary and compute reward

For more details on how dyna-q-learning works in general like update formula or planning, look into the algorithms in master thesis or read Sutton and Bartos book (Reinforcement Learning: An Introduction).

### Templates
- list for saving mean waveforms of clusters
- appending spike if new cluster is added
- otherwise computes mean template for every dimension of incoming spike
- mean is computed new-old with a learning rate to limit change of mean template

