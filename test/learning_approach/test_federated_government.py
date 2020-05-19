import numpy as np
from unittest.mock import Mock

from shfl.learning_approach.federated_government import FederatedGovernment
from shfl.data_base.data_base import DataBase
from shfl.data_distribution.data_distribution_iid import IidDataDistribution
from shfl.private.data import UnprotectedAccess
from shfl.private.federated_operation import split_train_test

class TestDataBase(DataBase):
    def __init__(self):
        super(TestDataBase, self).__init__()

    def load_data(self):
        self._train_data = np.random.rand(200).reshape([40, 5])
        self._test_data = np.random.rand(200).reshape([40, 5])
        self._validation_data = np.random.rand(200).reshape([40, 5])
        self._train_labels = np.random.randint(0, 10, 40)
        self._test_labels = np.random.randint(0, 10, 40)
        self._validation_labels = np.random.randint(0, 10, 40)


def test_evaluate_global_model():
    model_builder = Mock
    aggregator = Mock()
    database = TestDataBase()
    database.load_data()
    db = IidDataDistribution(database)

    num_nodes = 3
    federated_data, test_data, test_labels = db.get_federated_data(num_nodes)

    fdg = FederatedGovernment(model_builder, federated_data, aggregator)
    fdg._model.evaluate.return_value = np.random.randint(0, 10, 40)

    fdg.evaluate_global_model(test_data, test_labels)
    fdg._model.evaluate.assert_called_once_with(test_data, test_labels)


def test_deploy_central_model():
    model_builder = Mock
    aggregator = Mock()
    database = TestDataBase()
    database.load_data()
    db = IidDataDistribution(database)

    num_nodes = 3
    federated_data, test_data, test_labels = db.get_federated_data(num_nodes)

    fdg = FederatedGovernment(model_builder, federated_data, aggregator)
    array_params = np.random.rand(30)
    fdg._model.get_model_params.return_value = array_params

    fdg.deploy_central_model()

    for node in fdg._federated_data:
        node._model.set_model_params.assert_called_once_with(array_params)


def test_evaluate_clients_global():
    model_builder = Mock
    aggregator = Mock()
    database = TestDataBase()
    database.load_data()
    db = IidDataDistribution(database)

    num_nodes = 3
    federated_data, test_data, test_labels = db.get_federated_data(num_nodes)

    fdg = FederatedGovernment(model_builder, federated_data, aggregator)

    for node in fdg._federated_data:
        node.evaluate = Mock()
        node.evaluate.return_value = [np.random.randint(0, 10, 40), None]

    fdg.evaluate_clients(test_data, test_labels)

    for node in fdg._federated_data:
        node.evaluate.assert_called_once_with(test_data, test_labels)


def test_evaluate_clients_local():
    model_builder = Mock
    aggregator = Mock()
    database = TestDataBase()
    database.load_data()
    db = IidDataDistribution(database)

    num_nodes = 3
    federated_data, test_data, test_labels = db.get_federated_data(num_nodes)

    fdg = FederatedGovernment(model_builder, federated_data, aggregator)

    for node in fdg._federated_data:
        node.evaluate = Mock()
        node.evaluate.return_value = [np.random.randint(0, 10, 40), np.random.randint(0, 10, 40)]

    fdg.evaluate_clients(test_data, test_labels)

    for node in fdg._federated_data:
        node.evaluate.assert_called_once_with(test_data, test_labels)


def test_train_all_clients():
    model_builder = Mock
    aggregator = Mock()
    database = TestDataBase()
    database.load_data()
    db = IidDataDistribution(database)

    num_nodes = 3
    federated_data, test_data, test_labels = db.get_federated_data(num_nodes)

    fdg = FederatedGovernment(model_builder, federated_data, aggregator)

    fdg.train_all_clients()

    fdg._federated_data.configure_data_access(UnprotectedAccess())
    for node in fdg._federated_data:
        labeled_data = node.query()
        node._model.train.assert_called_once_with(labeled_data.data, labeled_data.label)


def test_aggregate_weights():
    model_builder = Mock
    aggregator = Mock()
    database = TestDataBase()
    database.load_data()
    db = IidDataDistribution(database)

    num_nodes = 3
    federated_data, test_data, test_labels = db.get_federated_data(num_nodes)

    fdg = FederatedGovernment(model_builder, federated_data, aggregator)

    weights = np.random.rand(64, 32)
    fdg._aggregator.aggregate_weights.return_value = weights

    fdg.aggregate_weights()

    fdg._model.set_model_params.assert_called_once_with(weights)


def test_run_rounds():
    model_builder = Mock
    aggregator = Mock()
    database = TestDataBase()
    database.load_data()
    db = IidDataDistribution(database)

    num_nodes = 3
    federated_data, test_data, test_labels = db.get_federated_data(num_nodes)

    fdg = FederatedGovernment(model_builder, federated_data, aggregator)

    fdg.deploy_central_model = Mock()
    fdg.train_all_clients = Mock()
    fdg.evaluate_clients = Mock()
    fdg.aggregate_weights = Mock()
    fdg.evaluate_global_model = Mock()

    fdg.run_rounds(1, test_data, test_labels)

    fdg.deploy_central_model.assert_called_once()
    fdg.train_all_clients.assert_called_once()
    fdg.evaluate_clients.assert_called_once_with(test_data, test_labels)
    fdg.aggregate_weights.assert_called_once()
    fdg.evaluate_global_model.assert_called_once_with(test_data, test_labels)


def test_run_rounds_local_tests():
    model_builder = Mock
    aggregator = Mock()
    database = TestDataBase()
    database.load_data()
    db = IidDataDistribution(database)

    num_nodes = 3
    federated_data, test_data, test_labels = db.get_federated_data(num_nodes)

    split_train_test(federated_data)

    fdg = FederatedGovernment(model_builder, federated_data, aggregator)

    fdg.deploy_central_model = Mock()
    fdg.train_all_clients = Mock()
    fdg.evaluate_clients = Mock()
    fdg.aggregate_weights = Mock()
    fdg.evaluate_global_model = Mock()

    fdg.run_rounds(1, test_data, test_labels)

    fdg.deploy_central_model.assert_called_once()
    fdg.train_all_clients.assert_called_once()
    fdg.evaluate_clients.assert_called_once_with(test_data, test_labels)
    fdg.aggregate_weights.assert_called_once()
    fdg.evaluate_global_model.assert_called_once_with(test_data, test_labels)