import logging
from unittest import TestCase
from _01_LoadDataset.LoadingDataset import LoadDataset
from _01_LoadDataset.SpikeClassToPytorchDataset import SpikeClassToPytorchDataset


class TestSpikeClassToPytorchDataset(TestCase):

    def setUp(self):
        self.path = "../_00_Datasets/03_SimDaten_Quiroga2020/004_C_Difficult1_noise005.mat"
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.dataset = LoadDataset(self.path, self.logger)
        self.dataloader, self.y_labels = self.dataset.loadData()

    def test_init(self):
        self.pytorch_dataloader = SpikeClassToPytorchDataset(self.dataloader.aligned_spikes, self.y_labels)

        self.assertEqual(self.pytorch_dataloader.data, self.dataloader.aligned_spikes)
        self.assertEqual(self.pytorch_dataloader.cluster.all(), self.y_labels.all())

    def test_length(self):
        self.pytorch_dataloader = SpikeClassToPytorchDataset(self.dataloader.aligned_spikes, self.y_labels)

        self.assertEqual(self.pytorch_dataloader.__len__(), len(self.y_labels))


