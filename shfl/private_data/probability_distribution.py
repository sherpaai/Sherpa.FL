import numpy as np
import abc


class ProbabilityDistribution(abc.ABC):
    """
    This class represents a probability distribution
    """

    @abc.abstractmethod
    def sample(self, size):
        """
        This method returns an array with length size sampling the distribution
        """


class NormalDistribution(ProbabilityDistribution):
    """
    Implement Normal Distribution
    """
    def __init__(self, mean, std):
        self._mean = mean
        self._std = std

    def sample(self, size):
        return np.random.normal(self._mean, self._std, size)


class GaussianMixture(ProbabilityDistribution):
    """
    Implement combination of Normal Distributions
    """
    def __init__(self, params, weights):
        self._gaussian_distributions = []
        for param in params:
            self._gaussian_distributions.append(NormalDistribution(param[0], param[1]))
        self._weights = weights

    def sample(self, size):
        mixture_idx = np.random.choice(len(self._weights), size=size, replace=True, p=self._weights)

        values = []
        for i in mixture_idx:
            gaussian_distributions = self._gaussian_distributions[i]
            values.append(gaussian_distributions.sample(1))

        return np.fromiter(values, dtype=np.float64)
