import numpy as np
from unittest.mock import Mock

from shfl.learning_approach.learning_approach import LearningApproach
from shfl.data_distribution.data_distribution_iid import IidDataDistribution
from shfl.data_base.data_base import DataBase


class TestLearningApproach(LearningApproach):
    def __init__(self, model_builder, federated_data, aggregator):
        super(TestLearningApproach, self).__init__(model_builder, federated_data, aggregator)
        
    def train_all_clients(self):
        pass

    def aggregate_weights(self):
        pass

    def run_rounds(self, n, test_data, test_label):
        pass


class TestDataBase(DataBase):
    def __init__(self):
        super(TestDataBase, self).__init__()

    def load_data(self):
        self._train_data = np.random.rand(50).reshape([10, 5])
        self._test_data = np.random.rand(50).reshape([10, 5])
        self._validation_data = np.random.rand(50).reshape([10, 5])
        self._train_labels = np.random.rand(10)
        self._test_labels = np.random.rand(10)
        self._validation_labels = np.random.rand(10)


def test_learning_approach_private_data():
    model_builder = Mock
    aggregator = Mock()
    database = TestDataBase()
    database.load_data()
    db = IidDataDistribution(database)
    federated_data, test_data, test_labels = db.get_federated_data("id001", 3)

    la = TestLearningApproach(model_builder, federated_data, aggregator)

    for node in la._federated_data:
        assert isinstance(node._model, model_builder)

    assert isinstance(la._model, model_builder)
    assert federated_data.identifier == la._federated_data.identifier
    assert aggregator.id == la._aggregator.id

