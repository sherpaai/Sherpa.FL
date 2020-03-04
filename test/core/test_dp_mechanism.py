import numpy as np
import pytest

import shfl
from shfl.core.data import DataAccessDefinition
from shfl.core.dp_mechanism import UnrandomizedMechanism
from shfl.core.dp_mechanism import RandomizeBinaryProperty
from shfl.core.dp_mechanism import LaplaceMechanism
from shfl.core.probability_distribution import NormalDistribution


def test_unrandomized_mechanism():
    data_size = 100
    array = np.random.rand(data_size)
    federated_array = shfl.core.federated_operation.federate_array("my_array", array, data_size)
    federated_array.configure_data_access(DataAccessDefinition(dp_mechanism=UnrandomizedMechanism()))
    for i in range(data_size):
        assert len(federated_array[i].query_private_data("my_array")) == 1
        assert federated_array[i].query_private_data("my_array") == array[i]


def test_randomize_binary_mechanism():
    data_size = 100
    array = np.ones(data_size)
    federated_array = shfl.core.federated_operation.federate_array("my_array", array, data_size)

    federated_array.configure_data_access(DataAccessDefinition(dp_mechanism=RandomizeBinaryProperty()))

    result = federated_array.query()
    differences = 0
    for i in range(data_size):
        if result[i] != array[i]:
            differences = differences + 1

    assert 0 < differences < data_size
    assert np.mean(result) < 1


def test_randomize_binary_mechanism_no_binary():
    data_size = 100
    array = np.random.rand(data_size)
    federated_array = shfl.core.federated_operation.federate_array("my_array", array, data_size)

    federated_array.configure_data_access(DataAccessDefinition(dp_mechanism=RandomizeBinaryProperty()))

    with pytest.raises(ValueError):
        federated_array.query()


def test_laplace_mechanism():
    data_size = 1000
    array = NormalDistribution(175, 7).sample(data_size)
    federated_array = shfl.core.federated_operation.federate_array("my_array", array, data_size)

    federated_array.configure_data_access(DataAccessDefinition(dp_mechanism=LaplaceMechanism(40, 1)))
    result = federated_array.query()

    differences = 0
    for i in range(data_size):
        if result[i] != array[i]:
            differences = differences + 1

    assert differences == data_size
    assert np.mean(array) - np.mean(result) < 5
