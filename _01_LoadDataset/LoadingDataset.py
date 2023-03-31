from _01_LoadDataset.ExternCode.load_mat_files import load_mat_file
from _01_LoadDataset.ExternCode.spike_class import spike_dataclass


class LoadDataset(object):
    def __init__(self, path, logger):
        self.path = path
        self.logger = logger

    def load_data(self):
        loaded_data = load_mat_file(self.path)

        data = dict()
        data['sampling_rate'] = float(1 / loaded_data['samplingInterval'][0][0] * 1000)
        data['raw_data'] = loaded_data['data'][0]
        data['spike_times'] = loaded_data['spike_times'][0][0][0]
        data['spike_cluster'] = loaded_data['spike_class'][0][0][0]

        dataloader = spike_dataclass(data)
        dataloader.align_spike_frames()
        self.logger.info(dataloader)
        self.logger.info(f"Sampling rate: {dataloader.sampling_rate}")
        self.logger.info(f"Raw: {dataloader.raw}")
        self.logger.info(f"Times: {dataloader.times}")
        self.logger.info(f"Cluster: {dataloader.cluster}")
        self.logger.info(f"Number of different clusters:  {max(dataloader.cluster)}")
        self.logger.info(f"Number of Spikes: {len(dataloader.cluster)}")
        self.logger.info(f"First aligned Spike Frame: {dataloader.aligned_spikes[0]}")

        y_labels = dataloader.cluster
        for i in range(0, max(dataloader.cluster)):
            y_labels[y_labels == i + 1] = i
            self.logger.info(f"Cluster {i}, Occurrences: {(y_labels == i).sum()}")

        return dataloader, y_labels
