from _01_LoadDataset.ExternCode.load_mat_files import load_mat_file
from _01_LoadDataset.ExternCode.spike_class import spike_dataclass


class LoadDataset:
    def loadData(self, path):
        loaded_data = load_mat_file(path)

        data = dict()
        data['sampling_rate'] = float(1 / loaded_data['samplingInterval'][0][0] * 1000)
        data['raw_data'] = loaded_data['data'][0]
        data['spike_times'] = loaded_data['spike_times'][0][0][0]
        data['spike_cluster'] = loaded_data['spike_class'][0][0][0]

        dataloader = spike_dataclass(data)
        dataloader.align_spike_frames()
        print(dataloader)
        print("Sampling rate: ", dataloader.sampling_rate)
        print("Raw: ", dataloader.raw)
        print("Times: ", dataloader.times)
        print("Cluster: ", dataloader.cluster)
        print("Number of different clusters: ", max(dataloader.cluster))
        print("Number of Spikes: ", len(dataloader.cluster))
        print("First aligned Spike Frame: ", dataloader.aligned_spikes[0])

        y_labels = dataloader.cluster
        for i in range(0, max(dataloader.cluster)):
            y_labels[y_labels == i + 1] = i
            print("Cluster ", i, " Occurences: ", (y_labels == i).sum())

        return dataloader, y_labels
