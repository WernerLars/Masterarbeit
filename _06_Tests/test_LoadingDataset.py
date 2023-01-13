from unittest import TestCase
import numpy as np
import logging

from _01_LoadDataset.LoadingDataset import LoadDataset


class TestLoadDataset(TestCase):

    def setUp(self):
        self.path = "../_00_Datasets/03_SimDaten_Quiroga2020/004_C_Difficult1_noise005.mat"
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.dataset = LoadDataset(self.path, self.logger)

    def test_load_data(self):
        dataloader, y_labels = self.dataset.loadData()
        self.assertEqual(np.ndarray, type(dataloader.raw))
        self.assertEqual(np.ndarray, type(dataloader.times))
        self.assertEqual(np.ndarray, type(dataloader.cluster))
        self.assertEqual(float, type(dataloader.sampling_rate))
        self.assertEqual(np.float64, type(dataloader.raw[0]))
        self.assertEqual(np.int32, type(dataloader.times[0]))
        self.assertEqual(np.uint8, type(dataloader.cluster[0]))

