import numpy as np
from unittest.mock import Mock
import pytest
from shfl.core.node import DataNode
from shfl.core.query import Get
from shfl.core.data import LabeledData


def test_private_data():
    random_array = np.random.rand(30)
    data_node = DataNode()
    data_node.set_private_data("random_array", random_array)
    data_node.private_data


def test_query_private_data():
    random_array = np.random.rand(30)
    data_node = DataNode()
    data_node.set_private_data("random_array", random_array)
    data = data_node.query_private_data(Get(), "random_array")
    for i in range(len(random_array)):
        assert data[i] == random_array[i]


def test_query_model_params():
    random_array = np.random.rand(30)
    data_node = DataNode()
    model_mock = Mock()
    model_mock.get_model_params.return_value = random_array
    data_node.model = model_mock
    model_params = data_node.query_model_params(Get())
    for i in range(len(random_array)):
        assert model_params[i] == random_array[i]


def test_train_model_wrong_data():
    random_array = np.random.rand(30)
    data_node = DataNode()
    model_mock = Mock()
    data_node.model = model_mock
    data_node.set_private_data("random_array", random_array)
    with pytest.raises(ValueError):
        data_node.train_model("random_array")


def test_train_model_data():
    random_array = np.random.rand(30)
    random_array_labels = np.random.rand(30)
    labeled_data = LabeledData(random_array, random_array_labels)
    data_node = DataNode()
    model_mock = Mock()
    data_node.model = model_mock
    data_node.set_private_data("random_array", labeled_data)
    data_node.train_model("random_array")
    model_mock.train.assert_called_once()


def test_get_model():
    model_mock = Mock()
    data_node = DataNode()
    data_node.model = model_mock
    assert data_node.model is None


def test_predict():
    random_array = np.random.rand(30)
    model_mock = Mock()
    data_node = DataNode()
    data_node.model = model_mock
    data_node.predict(random_array)
    model_mock.predict.assert_called_once_with(random_array)


def test_set_params():
    random_array = np.random.rand(30)
    model_mock = Mock()
    data_node = DataNode()
    data_node.model = model_mock
    data_node.set_model_params(random_array)
    model_mock.set_model_params.assert_called_once_with(random_array)
