import numpy as np
from unittest.mock import Mock
import pytest

from shfl.private.node import DataNode
from shfl.private.data import LabeledData
from shfl.private.data import UnprotectedAccess


def test_private_data():
    random_array = np.random.rand(30)
    data_node = DataNode()
    data_node.set_private_data("random_array", random_array)
    data_node.private_data


def test_private_test_data():
    random_array = np.random.rand(30)
    data_node = DataNode()
    data_node.set_private_test_data("random_array_test", random_array)
    data_node.private_test_data


def test_query_private_data():
    random_array = np.random.rand(30)
    data_node = DataNode()
    data_node.set_private_data("random_array", random_array)
    data_node.configure_data_access("random_array", UnprotectedAccess())
    data = data_node.query("random_array")
    for i in range(len(random_array)):
        assert data[i] == random_array[i]


def test_query_model_params():
    random_array = np.random.rand(30)
    data_node = DataNode()
    model_mock = Mock()
    model_mock.get_model_params.return_value = random_array
    data_node.model = model_mock
    data_node.configure_model_params_access(UnprotectedAccess())
    model_params = data_node.query_model_params()
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


def test_evaluate():
    data = np.random.rand(60).reshape((15, 4))
    labels = np.random.randint(0, 2, 15)

    data_node = DataNode()
    data_node._model = Mock()

    data_node.evaluate(data, labels)

    data_node._model.evaluate.assert_called_once_with(data, labels)


def test_local_evaluate():
    data_key = 'EMNIST'
    data_node = DataNode()
    data_node._private_test_data = Mock()

    data = Mock()
    data.data = np.random.rand(60)
    data.label = np.random.randint(0, 2, 60)
    data_node._private_test_data.get.return_value = data

    data_node._model = Mock()
    data_node._model.evaluate.return_value = 0

    data_node.self_private_test_data = 1

    eval = data_node.local_evaluate(data_key)

    assert eval == 0
    data_node._private_test_data.get.assert_called_once_with(data_key)
    data_node._model.evaluate.assert_called_once_with(data.data, data.label)


def test_local_evaluate_wrong():
    data_node = DataNode()
    data_node.self_private_test_data = 0

    eval = data_node.local_evaluate('id')

    assert eval is None
