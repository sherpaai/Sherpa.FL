import abc
from math import log
from math import exp
from scipy.special import comb
import numpy as np


class Sampler():
    """
    This class implements sampling methods which helps to reduce
    the epsilon-delta budget spent by a dp-mechanism

    # Arguments:
        sample_size: size of the sample to be taken
    """

    @abc.abstractmethod
    def epsilon_delta_reduction(self, epsilon_delta):
        """
        It receives epsilon_delta parameters from a dp-mechanism
        and computes the new hopefully reduced epsilon_delta

        # Arguments:
            epsilon_delta: privacy budget provided by a dp-mechanism

        # Returns
            new_epsilon_delta: new hopefully reduced epsilon_delta
        """

    @abc.abstractmethod
    def sample(self, data):
        """
        It receives some data and returns a sample of it

        # Arguments:
            data: Raw data that are going to be sampled

        # Returns
            sampled_data: sample of size self._sample_size
        """


class DefaultSampler(Sampler):
    """
        Default sampler implementation
    """
    def __init__(self):
        pass

    def sample(self, data):
        return data

    def epsilon_delta_reduction(self, epsilon_delta):
        return epsilon_delta


class SampleWithReplacement(Sampler):
    """
        It implements the sample with replacement technique (Theorem 10 from the reference) which reduces
        the epsilon-delta bugdet spent specified
    
        # Arguments:
            sample_size: size of the sample
            data_size: size of the input data
        
        # References:
            - [Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences](https://arxiv.org/abs/1807.01647)
    """

    def __init__(self, sample_size, data_size):
        check_sample_size(sample_size, data_size)
        self._sample_size = sample_size
        self._data_size = data_size

    def sample(self, data):
        return np.random.choice(data, self._sample_size, replace=True)

    def epsilon_delta_reduction(self, epsilon_delta):
        proportion = 1 - (1 - 1/self._data_size)**self._sample_size
        epsilon, delta = epsilon_delta

        new_epsilon = log(1+proportion*(exp(epsilon)-1))
        n = self._data_size
        m = self._sample_size
        new_delta = sum([comb(m, k) * ((1 / n) ** k) * ((1 - 1 / n) ** (m - k)) for k in range(1, m+1)])
        new_delta *= delta

        return new_epsilon, new_delta


class SampleWithoutReplacement(Sampler):
    """
        It implements the sample with replacement technique (Theorem 9 from the reference) which reduces
        the epsilon-delta bugdet spent specified
    
        # Arguments:
            sample_size: size of the sample
            data_size: size of the input data
        
        # References:
            - [Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences](https://arxiv.org/abs/1807.01647)
    """

    def __init__(self, sample_size, data_size):
        check_sample_size(sample_size, data_size)
        self._sample_size = sample_size
        self._data_size = data_size

    def sample(self, data):
        return np.random.choice(data, self._sample_size, replace=False)

    def epsilon_delta_reduction(self, epsilon_delta):
        proportion = self._sample_size/self._data_size
        epsilon, delta = epsilon_delta

        new_epsilon = log(1+proportion*(exp(epsilon)-1))
        new_delta = proportion*delta

        return new_epsilon, new_delta


def check_sample_size(sample_size, data_size):
    if sample_size > data_size:
        raise ValueError("Sample size {} must be less than data size: {}".format(sample_size, data_size))
