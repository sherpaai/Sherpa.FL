import numpy as np
import pandas as pd
import pytest
from tensorflow.keras.utils import to_categorical
from unittest.mock import patch

import shfl.private.federated_operation
from shfl.private.federated_operation import FederatedTransformation
from shfl.private.federated_operation import FederatedData
from shfl.private.federated_operation import FederatedDataNode
from shfl.private.data import UnprotectedAccess, LabeledData


class TestTransformation(FederatedTransformation):
    def apply(self, data):
        data += 1


def test_federate_transformation():
    random_array = np.random.rand(30)
    federated_array = shfl.private.federated_operation.federate_array(random_array, 30)
    federated_array.configure_data_access(UnprotectedAccess())
    shfl.private.federated_operation.apply_federated_transformation(federated_array, TestTransformation())
    index = 0
    for data_node in federated_array:
        assert data_node.query() == random_array[index] + 1
        index = index + 1


def test_query_federate_data():
    random_array = np.random.rand(30)
    federated_array = shfl.private.federated_operation.federate_array(random_array, 30)
    federated_array.configure_data_access(UnprotectedAccess())
    answer = federated_array.query()
    for i in range(len(answer)):
        assert answer[i] == random_array[i]


def test_federate_array():
    data_size = 10000
    num_clients = 1000
    array = np.random.rand(data_size)
    federated_array = shfl.private.federated_operation.federate_array(array, num_clients)
    assert federated_array.num_nodes() == num_clients


def test_federate_array_size_private_data():
    data_size = 10000
    num_clients = 10
    array = np.random.rand(data_size)
    federated_array = shfl.private.federated_operation.federate_array(array, num_clients)
    federated_array.configure_data_access(UnprotectedAccess())
    for data_node in federated_array:
        assert len(data_node.query()) == data_size/num_clients

    assert federated_array[0].query()[0] == array[0]


def test_federated_data():
    data_size = 10
    federated_data = FederatedData()
    assert federated_data.num_nodes() == 0
    array = np.random.rand(data_size)
    federated_data.add_data_node(array)
    federated_data.configure_data_access(UnprotectedAccess())
    assert federated_data.num_nodes() == 1
    assert federated_data[0].query()[0] == array[0]


def test_federated_data_identifier():
    data_size = 10
    federated_data = FederatedData()
    array = np.random.rand(data_size)
    federated_data.add_data_node(array)
    federated_data.configure_data_access(UnprotectedAccess())
    with pytest.raises(ValueError):
        federated_data[0].query("bad_identifier_federated_data")


def test_split_train_test():
    num_nodes = 10
    data = np.random.rand(10,num_nodes)
    label = np.random.randint(range(0, 10), num_nodes)

    federated_data = FederatedData()
    for idx in range(num_nodes):
        federated_data.add_data_node(LabeledData(data[idx], to_categorical(label[idx])))

    federated_data.configure_data_access(UnprotectedAccess())
    raw_federated_data = federated_data

    shfl.private.federated_operation.split_train_test(federated_data)

    for raw_node, split_node in zip(raw_federated_data, federated_data):
        raw_node.split_train_test()
        assert raw_node.private_data == split_node.private_data
        assert raw_node.private_test_data == split_node.private_test_data


def test_split_train_test_pandas():
    num_nodes = 10
    data = pd.DataFrame(np.random.rand(10, num_nodes))
    label = pd.Series(np.random.randint(range(0, 10), num_nodes))

    federated_data = FederatedData()
    for idx in range(num_nodes):
        federated_data.add_data_node(LabeledData(data[idx], to_categorical(label[idx])))

    federated_data.configure_data_access(UnprotectedAccess())
    raw_federated_data = federated_data

    shfl.private.federated_operation.split_train_test(federated_data)

    for raw_node, split_node in zip(raw_federated_data, federated_data):
        raw_node.split_train_test()
        assert raw_node.private_data == split_node.private_data
        assert raw_node.private_test_data == split_node.private_test_data


@patch("shfl.private.federated_operation.DataNode.evaluate")
@patch("shfl.private.federated_operation.DataNode.local_evaluate")
def test_evaluate(mock_super_local_evaluate, mock_super_evaluate):
    data = np.random.rand(15)
    test = np.random.rand(5)

    identifier = 'id'
    fdn = FederatedDataNode(identifier)

    mock_super_evaluate.return_value = 10
    mock_super_local_evaluate.return_value = 15

    evaluate, local_evaluate = fdn.evaluate(data, test)

    assert evaluate == 10
    assert local_evaluate == 15

    mock_super_evaluate.assert_called_once_with(data, test)
    mock_super_local_evaluate.assert_called_once_with(identifier)


