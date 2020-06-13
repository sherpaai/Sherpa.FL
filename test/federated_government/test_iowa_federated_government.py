from unittest.mock import Mock

import numpy as np
import pytest

from shfl.data_base.data_base import DataBase
from shfl.data_distribution.data_distribution_iid import IidDataDistribution
from shfl.federated_aggregator.iowa_federated_aggregator import IowaFederatedAggregator
from shfl.federated_government.iowa_federated_government import IowaFederatedGovernment


class TestDataBase(DataBase):
    def __init__(self):
        super(TestDataBase, self).__init__()

    def load_data(self):
        self._train_data = np.random.rand(50).reshape([10, 5])
        self._test_data = np.random.rand(50).reshape([10, 5])
        self._train_labels = np.random.randint(0, 2, 10)
        self._test_labels = np.random.randint(0, 2, 10)


def test_IowaFederatedGovernment():
    model_builder = Mock
    database = TestDataBase()
    database.load_data()
    db = IidDataDistribution(database)

    num_nodes = 3
    federated_data, test_data, test_labels = db.get_federated_data(num_nodes)

    a = 0
    b = 1
    c = 2
    y_b = 3
    k = 4
    dynamic = True
    iowa_fg = IowaFederatedGovernment(model_builder, federated_data, model_params_access=None,
                                      dynamic=dynamic, a=a, b=b, c=c, y_b=y_b, k=k)

    assert isinstance(iowa_fg._aggregator, IowaFederatedAggregator)
    assert isinstance(iowa_fg._model, model_builder)
    assert np.array_equal(iowa_fg._federated_data, federated_data)
    assert iowa_fg._a == a
    assert iowa_fg._b == b
    assert iowa_fg._c == c
    assert iowa_fg._y_b == y_b
    assert iowa_fg._k == k
    assert iowa_fg._dynamic == dynamic


def test_performance_clients():
    model_builder = Mock
    database = TestDataBase()
    database.load_data()
    db = IidDataDistribution(database)

    num_nodes = 3
    federated_data, test_data, test_labels = db.get_federated_data(num_nodes)

    iowa_fg = IowaFederatedGovernment(model_builder, federated_data)
    for i, data_node in enumerate(iowa_fg._federated_data):
        data_node.performance = Mock()
        data_node.performance.return_value = i
    res = np.arange(iowa_fg._federated_data.num_nodes())

    data_val = np.random.rand(25).reshape((5, 5))
    labels_val = np.random.randint(0, 2, 5)
    performance = iowa_fg.performance_clients(data_val, labels_val)

    assert np.array_equal(performance, res)
    for data_node in iowa_fg._federated_data:
        data_node.performance.assert_called_once_with(data_val, labels_val)


def test_run_rounds():
    np.random.seed(123)
    model_builder = Mock
    database = TestDataBase()
    database.load_data()
    db = IidDataDistribution(database)

    num_nodes = 3
    federated_data, test_data, test_label = db.get_federated_data(num_nodes)

    iowa_fg = IowaFederatedGovernment(model_builder, federated_data)

    n = 1

    iowa_fg.deploy_central_model = Mock()
    iowa_fg.train_all_clients = Mock()
    iowa_fg.evaluate_clients = Mock()
    iowa_fg.performance_clients = Mock()
    iowa_fg.performance_clients.return_value = 0
    iowa_fg._aggregator.set_ponderation = Mock()
    iowa_fg.aggregate_weights = Mock()
    iowa_fg.evaluate_global_model = Mock()

    iowa_fg.run_rounds(n, test_data, test_label)
    # Replicate test an validate data
    randomize = [0, 9, 3, 4, 6, 8, 2, 1, 5, 7]
    test_data = test_data[randomize,]
    test_label = test_label[randomize]
    validation_data = test_data[:int(0.15 * len(test_label)), ]
    validation_label = test_label[:int(0.15 * len(test_label))]
    test_data = test_data[int(0.15 * len(test_label)):, ]
    test_label = test_label[int(0.15 * len(test_label)):]

    iowa_fg.deploy_central_model.assert_called_once()
    iowa_fg.deploy_central_model.assert_called_once()
    iowa_fg.train_all_clients.assert_called_once()
    iowa_fg.evaluate_clients.assert_called_once()
    assert len(iowa_fg.evaluate_clients.call_args[0]) == 2
    np.testing.assert_array_equal(iowa_fg.evaluate_clients.call_args[0][0], test_data)
    np.testing.assert_array_equal(iowa_fg.evaluate_clients.call_args[0][1], test_label)
    iowa_fg.performance_clients.assert_called_once()
    assert len(iowa_fg.performance_clients.call_args[0]) == 2
    np.testing.assert_array_equal(iowa_fg.performance_clients.call_args[0][0], validation_data)
    np.testing.assert_array_equal(iowa_fg.performance_clients.call_args[0][1], validation_label)
    iowa_fg._aggregator.set_ponderation.assert_called_once_with(iowa_fg.performance_clients.return_value,
                                                                iowa_fg._dynamic, iowa_fg._a, iowa_fg._b, iowa_fg._c,
                                                                iowa_fg._y_b,
                                                                iowa_fg._k)
    iowa_fg.aggregate_weights.assert_called_once()
    iowa_fg.evaluate_global_model.assert_called_once()
    assert len(iowa_fg.evaluate_global_model.call_args[0]) == 2
    np.testing.assert_array_equal(iowa_fg.evaluate_clients.call_args[0][0], test_data)
    np.testing.assert_array_equal(iowa_fg.evaluate_global_model.call_args[0][1], test_label)
