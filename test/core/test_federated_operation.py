import numpy as np
import pytest

import shfl.private_data.federated_operation
from shfl.private_data.node import DataNode
from shfl.private_data.query import IdentityFunction
from shfl.private_data.federated_operation import FederatedTransformation
from shfl.private_data.federated_operation import FederatedData
from shfl.private_data.data import UnprotectedAccess


class TestTransformation(FederatedTransformation):
    def apply(self, data):
        data += 1


def test_federate_transformation():
    random_array = np.random.rand(30)
    federated_array = shfl.private_data.federated_operation.federate_array("my_federated_array", random_array, 30)
    federated_array.configure_data_access(UnprotectedAccess())
    shfl.private_data.federated_operation.apply_federated_transformation(federated_array, TestTransformation())
    index = 0
    for data_node in federated_array:
        assert data_node.query_private_data("my_federated_array") == random_array[index] + 1
        index = index + 1


def test_query_federate_data():
    random_array = np.random.rand(30)
    federated_array = shfl.private_data.federated_operation.federate_array("my_federated_array", random_array, 30)
    federated_array.configure_data_access(UnprotectedAccess())
    answer = federated_array.query()
    for i in range(len(answer)):
        assert answer[i] == random_array[i]


def test_federate_array():
    data_size = 10000
    num_clients = 1000
    array = np.random.rand(data_size)
    federated_array = shfl.private_data.federated_operation.federate_array("my_array1", array, num_clients)
    assert federated_array.num_nodes() == num_clients
    assert federated_array.identifier == "my_array1"


def test_federate_array_size_private_data():
    data_size = 10000
    num_clients = 10
    array = np.random.rand(data_size)
    federated_array = shfl.private_data.federated_operation.federate_array("my_array", array, num_clients)
    federated_array.configure_data_access(UnprotectedAccess())
    for data_node in federated_array:
        assert len(data_node.query_private_data("my_array")) == data_size/num_clients

    assert federated_array[0].query_private_data("my_array")[0] == array[0]


def test_federated_data():
    data_size = 10
    federated_data = FederatedData("my_federated_data")
    assert federated_data.num_nodes() == 0
    array = np.random.rand(data_size)
    federated_data.add_data_node(DataNode(), array)
    federated_data.configure_data_access(UnprotectedAccess())
    assert federated_data.num_nodes() == 1
    assert federated_data[0].query_private_data("my_federated_data")[0] == array[0]


def test_federated_data_identifier():
    data_size = 10
    federated_data = FederatedData("my_federated_data")
    array = np.random.rand(data_size)
    federated_data.add_data_node(DataNode(), array)
    federated_data.configure_data_access(UnprotectedAccess())
    with pytest.raises(ValueError):
        federated_data[0].query_private_data("bad_identifier_federated_data")


def test_federated_data_repeated_identifier():
    my_data = FederatedData("my_federated_data")
    with pytest.raises(ValueError):
        FederatedData("my_federated_data")

