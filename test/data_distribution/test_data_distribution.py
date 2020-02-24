import numpy as np
import pytest

from shfl.data_base.data_base import DataBase
from shfl.data_distribution.data_distribution import DataDistribution
from shfl.core.query import Get


class TestDataDistribution(DataDistribution):
    def __init__(self,database):
        super(TestDataDistribution, self).__init__(database)

    def make_data_federated(self, data, labels, num_nodes, percent, weights):
        pass

class TestDataBase(DataBase):
    def __init__(self):
        super(TestDataBase, self).__init__()

    def load_data(self):
        self._train_data = np.random.rand(50).reshape([10,5])
        self._test_data = np.random.rand(50).reshape([10,5])
        self._validation_data = np.random.rand(50).reshape([10,5])
        self._train_labels = np.random.randint(0,10,10)
        self._test_labels = np.random.randint(0,10,10)
        self._validation_labels = np.random.randint(0,10,10)


def test_data_distribution_private_data():
    data = TestDataBase()
    data.load_data()

    dt = TestDataDistribution(data)

    assert data == dt._database


