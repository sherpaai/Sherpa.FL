import numpy as np
import pytest

from math import log
from math import exp
from shfl.private import DataNode
from shfl.differential_privacy.dp_mechanism import LaplaceMechanism
from shfl.differential_privacy.dp_mechanism import GaussianMechanism
from shfl.differential_privacy.dp_mechanism import RandomizedResponseBinary
from shfl.differential_privacy.dp_mechanism import RandomizedResponseCoins
from shfl.differential_privacy.dp_mechanism import ExponentialMechanism
from shfl.differential_privacy.dp_sampling import SampleWithoutReplacement
from shfl.differential_privacy.dp_sampling import SampleWithReplacement


def test_sample_with_replacement():
    array = np.ones(100)
    node_single = DataNode()
    node_single.set_private_data(name="array", data=array)
    sample_size = 50
    sampling_method = SampleWithReplacement(sample_size, 100)
    
    def u(x, r):
        output = np.zeros(len(r))
        for i in range(len(r)):
            output[i] = r[i] * sum(np.greater_equal(x, r[i]))
        return output

    r = np.arange(0, 3.5, 0.001)
    delta_u = r.max()
    epsilon = 5
    exponential_mechanism = ExponentialMechanism(u, r, delta_u, epsilon, size=sample_size)
    
    access_modes = [LaplaceMechanism(1, 1, sampling_method=sampling_method)]
    access_modes.append(GaussianMechanism(1, (0.5, 0.5), sampling_method=sampling_method))
    access_modes.append(RandomizedResponseBinary(0.5, 0.5, 1, sampling_method=sampling_method))
    access_modes.append(RandomizedResponseCoins(sampling_method=sampling_method))
    access_modes.append(exponential_mechanism)
    
    for a in access_modes:
        node_single.configure_data_access("array", a)
        result = node_single.query("array")
        assert result.shape[0] == sample_size


def test_sample_without_replacement():
    array = np.ones(100)
    node_single = DataNode()
    node_single.set_private_data(name="array", data=array)
    sample_size = 50
    sampling_method = SampleWithoutReplacement(sample_size, 100)
    
    def u(x, r):
        output = np.zeros(len(r))
        for i in range(len(r)):
            output[i] = r[i] * sum(np.greater_equal(x, r[i]))
        return output

    r = np.arange(0, 3.5, 0.001)
    delta_u = r.max()
    epsilon = 5
    exponential_mechanism = ExponentialMechanism(u, r, delta_u, epsilon, size=sample_size)
    
    access_modes = [LaplaceMechanism(1, 1, sampling_method=sampling_method)]
    access_modes.append(GaussianMechanism(1, (0.5, 0.5), sampling_method=sampling_method))
    access_modes.append(RandomizedResponseBinary(0.5, 0.5, 1, sampling_method=sampling_method))
    access_modes.append(RandomizedResponseCoins(sampling_method=sampling_method))
    access_modes.append(exponential_mechanism)
    
    for a in access_modes:
        node_single.configure_data_access("array", a)
        result = node_single.query("array")
        assert result.shape[0] == sample_size
    
def test_sample_error():
    array = np.ones(100)
    node_single = DataNode()
    node_single.set_private_data(name="array", data=array)
    sample_size = 101
    
    with pytest.raises(ValueError):
        access_mode = LaplaceMechanism(1, 1, sampling_method=SampleWithoutReplacement(sample_size, 100))
        
    with pytest.raises(ValueError):
        access_mode = LaplaceMechanism(1, 1, sampling_method=SampleWithReplacement(sample_size, 100))
        
        
def test_epsilon_delta_reduction():
    epsilon = 1
    n = 100
    m = 50
    sampling_method = SampleWithoutReplacement(m, n)
    access_mode = LaplaceMechanism(1, epsilon, sampling_method=sampling_method)
    proportion = m/n
    assert access_mode.epsilon_delta == (log(1+proportion*(exp(epsilon)-1)), 0)
    
    sampling_method = SampleWithReplacement(m, n)
    access_mode = LaplaceMechanism(1, epsilon, sampling_method=sampling_method)
    proportion = 1 - (1 - 1 / n) ** m
    assert access_mode.epsilon_delta == (log(1 + proportion * (exp(epsilon) - 1)), 0)
