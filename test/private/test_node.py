import numpy as np
from unittest.mock import Mock
import pytest

from shfl.private.node import DataNode
from shfl.private.data import LabeledData
from shfl.private.data import UnprotectedAccess
from shfl.differential_privacy.dp_mechanism import GaussianMechanism
from shfl.private import ExceededPrivacyBudgetError

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
    
def test_exception_exceededprivacybudgeterror():
    scalar = 175

    node = DataNode(epsilon_delta=(1, 0))
    node.set_private_data("scalar", scalar)
    node.configure_data_access("scalar", GaussianMechanism(1, epsilon_delta=(0.1, 1)))

    with pytest.raises(ExceededPrivacyBudgetError):
        node.query("scalar")
        
    try:
        node.query("scalar")
    except ExceededPrivacyBudgetError as e:
        assert str(e) == 'Error: Privacy Budget {} has been exceeded for property {}'.format((1,0), "scalar")
        
    try:
        raise ExceededPrivacyBudgetError(epsilon_delta=(0, 1))
    except ExceededPrivacyBudgetError as e:
        assert str(e) == 'Error: Privacy Budget {} has been exceeded'.format((0,1))

    try:
        raise ExceededPrivacyBudgetError()
    except ExceededPrivacyBudgetError as e:
        assert str(e) == 'Error: Privacy Budget has been exceeded'

    try:
        raise ExceededPrivacyBudgetError(property="scalar")
    except ExceededPrivacyBudgetError as e:
        assert str(e) == 'Error: Privacy Budget has been exceeded for property scalar'
    
def test_constructor_bad_params():
    with pytest.raises(ValueError):
        DataNode(epsilon_delta=(1,2,3))

    with pytest.raises(ValueError):
        DataNode(epsilon_delta=(-1,2))
        
    with pytest.raises(ValueError):
        DataNode(epsilon_delta=(1,-2))
        
def test_get_epsilon_delta():
    e_d = (1, 1)
    data_node = DataNode(epsilon_delta=e_d)
    
    assert data_node.epsilon_delta == e_d

def test_configure_data_access():
    data_node = DataNode()
    data_node.set_private_data("test", np.array(range(10)))
    with pytest.raises(ValueError):
        data_node.configure_data_access("test", GaussianMechanism(1, epsilon_delta=(0.1,1)))

def test_configure_data_access2():
    data_node = DataNode(epsilon_delta=(1,1))
    data_node.set_private_data("test", np.array(range(10)))
    with pytest.raises(ValueError):
        data_node.configure_data_access("test", UnprotectedAccess())