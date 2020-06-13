from shfl.federated_aggregator.iowa_federated_aggregator import IowaFederatedAggregator
from unittest.mock import Mock, patch
import pytest
import numpy as np


@patch('shfl.federated_aggregator.iowa_federated_aggregator.super')
def test_IowaFederatedAggregator(mock_super):
    iowa = IowaFederatedAggregator()

    mock_super.assert_called_once()
    assert iowa._a == 0
    assert iowa._b == 0
    assert iowa._c == 0
    assert iowa._y_b == 0
    assert iowa._k == 0
    assert iowa._performance is None
    assert iowa._dynamic is None


def test_set_ponderation():
    a = 0
    b = 1
    c = 2
    y_b = 3
    k = 4
    performance = True
    dynamic = False

    iowa = IowaFederatedAggregator()
    iowa.get_ponderation_weights = Mock()

    iowa.set_ponderation(performance, dynamic, a, b, c, y_b, k)

    iowa.get_ponderation_weights.assert_called_once()

    assert iowa._a == a
    assert iowa._b == b
    assert iowa._c == c
    assert iowa._y_b == y_b
    assert iowa._k == k
    assert iowa._performance
    assert not iowa._dynamic


def test_set_ponderation_wrong_performance():
    performance = [1, 2, 3]

    iowa = IowaFederatedAggregator()

    with pytest.raises(TypeError):
        iowa.set_ponderation(performance)


def test_q_function():
    iowa = IowaFederatedAggregator()
    iowa.get_ponderation_weights = Mock()
    iowa.set_ponderation(None)
    res = iowa.q_function(0)

    assert res == 0

    res = iowa.q_function(0.2)

    assert res == (0.2 - iowa._a) / (iowa._b - iowa._a) * iowa._y_b

    res = iowa.q_function(0.8)

    assert res == (0.8 - iowa._b) / (iowa._c - iowa._b) * (1 - iowa._y_b) + iowa._y_b

    res = iowa.q_function(10)

    assert res == 1


def test_get_ponderation_weights_not_dynamic():
    performance = np.array([1, 5, 2])
    res = [0.53, 0.33, 0.13]
    iowa = IowaFederatedAggregator()
    iowa.set_ponderation(performance=performance, dynamic=False)

    weights = iowa.get_ponderation_weights()

    assert np.array_equal(iowa._performance, np.sort(performance)[::-1])
    assert pytest.approx(res, weights, 0.01)


def test_get_ponderation_weights():
    performance = np.array([1, 5, 2])
    res = [0.5, 0.25, 0.25]
    iowa = IowaFederatedAggregator()
    iowa.set_ponderation(performance=performance)

    weights = iowa.get_ponderation_weights()

    assert np.array_equal(iowa._performance, np.sort(performance)[::-1])
    assert np.array_equal(res, weights)


def test_get_ponderation_weights_wrong_performance():
    iowa = IowaFederatedAggregator()

    with pytest.raises(TypeError):
        iowa.get_ponderation_weights()
