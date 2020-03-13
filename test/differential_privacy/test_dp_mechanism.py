import numpy as np
import pytest

import shfl
from shfl.private import DataNode
from shfl.private.data import DataAccessDefinition
from shfl.differential_privacy.dp_mechanism import UnrandomizedMechanism, RandomizedResponseBinary
from shfl.differential_privacy.dp_mechanism import RandomizeBinaryProperty
from shfl.differential_privacy.dp_mechanism import LaplaceMechanism
from shfl.differential_privacy.probability_distribution import NormalDistribution


def test_unrandomized_mechanism():
    data_size = 100
    array = np.random.rand(data_size)
    federated_array = shfl.private.federated_operation.federate_array("my_array", array, data_size)
    federated_array.configure_data_access(DataAccessDefinition(dp_mechanism=UnrandomizedMechanism()))
    for i in range(data_size):
        assert len(federated_array[i].query_private_data("my_array")) == 1
        assert federated_array[i].query_private_data("my_array") == array[i]


def test_randomize_binary_mechanism():
    data_size = 100
    array = np.ones(data_size)
    federated_array = shfl.private.federated_operation.federate_array("my_array", array, data_size)

    federated_array.configure_data_access(DataAccessDefinition(dp_mechanism=RandomizeBinaryProperty()))

    result = federated_array.query()
    differences = 0
    for i in range(data_size):
        if result[i] != array[i]:
            differences = differences + 1

    assert 0 < differences < data_size
    assert np.mean(result) < 1

def test_randomize_binary_deterministic():
    array = np.array([0, 1])
    node_single = DataNode()
    node_single.set_private_data(name="A", data=array)
    dp_mechanism = RandomizedResponseBinary(f0=1, f1=1)
    data_access_definition = DataAccessDefinition(dp_mechanism=dp_mechanism)
    node_single.configure_private_data_access("A", data_access_definition)

    result = node_single.query_private_data(private_property="A")

    assert np.array_equal(array, result)


def test_randomize_binary_mechanism_no_binary():
    data_size = 100
    array = np.random.rand(data_size)
    federated_array = shfl.private.federated_operation.federate_array("my_array", array, data_size)

    federated_array.configure_data_access(DataAccessDefinition(dp_mechanism=RandomizeBinaryProperty()))

    with pytest.raises(ValueError):
        federated_array.query()


def test_laplace_mechanism():
    data_size = 1000
    array = NormalDistribution(175, 7).sample(data_size)
    federated_array = shfl.private.federated_operation.federate_array("my_array", array, data_size)

    federated_array.configure_data_access(DataAccessDefinition(dp_mechanism=LaplaceMechanism(40, 1)))
    result = federated_array.query()

    differences = 0
    for i in range(data_size):
        if result[i] != array[i]:
            differences = differences + 1

    assert differences == data_size
    assert np.mean(array) - np.mean(result) < 5
