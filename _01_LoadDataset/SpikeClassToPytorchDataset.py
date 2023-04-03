from torch.utils.data import Dataset
import torch


class SpikeClassToPytorchDataset(Dataset):
    def __init__(self, aligned_spikes, cluster):
        self.data = aligned_spikes
        self.cluster = cluster

    def __len__(self):
        return len(self.cluster)

    def __getitem__(self, idx):
        """
        :param idx: position in data (see custom dataloader in pytorch)
        :return: spike and cluster label
        """

        spike_window = torch.tensor(self.data[idx], dtype=torch.float)
        cluster = self.cluster[idx]
        return spike_window, cluster
