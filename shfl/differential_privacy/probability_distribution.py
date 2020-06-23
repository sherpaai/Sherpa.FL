import numpy as np
import abc


class ProbabilityDistribution(abc.ABC):
    """
    Class representing the interface for a probability distribution
    """

    @abc.abstractmethod
    def sample(self, size):
        """
        This method must return an array with length "size", sampling the distribution

        # Arguments:
            size: Size of the sampling
        """


class NormalDistribution(ProbabilityDistribution):
    """
    Implements Normal Distribution

    # Arguments:
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution
    """
    def __init__(self, mean, std):
        self._mean = mean
        self._std = std

    def sample(self, size):
        """
        This method provides a sample of the given size of a gaussian distributions

        # Arguments:
            size: size of the sample

        # Returns:
            Sample of a gaussian distribution of a given size
        """
        return np.random.normal(self._mean, self._std, size)


class GaussianMixture(ProbabilityDistribution):
    """
    Implements the combination of Normal Distributions

    # Arguments:
        params: Array of arrays with mean and std for every gaussian distribution.
        weights: Array of weights for every distribution with sum 1.

    # Example:

    ```python
        # Parameters for two Gaussian
        mu_M = 178
        mu_F = 162
        sigma_M = 7
        sigma_F = 7

        # Parameters
        norm_params = np.array([[mu_M, sigma_M],
                               [mu_F, sigma_F]])
        weights = np.ones(2) / 2.0

        # Creating combination of gaussian
        distribution = GaussianMixture(norm_params, weights)
    ```
    """
    def __init__(self, params, weights):
        self._gaussian_distributions = []
        for param in params:
            self._gaussian_distributions.append(NormalDistribution(param[0], param[1]))
        self._weights = weights

    def sample(self, size):
        """
        This method provides a sample of the given size of a mixture of gaussian distributions

        # Arguments:
            size: size of the sample

        # Returns:
            Sample of a mixture of gaussian distributions of a given size
        """

        mixture_idx = np.random.choice(len(self._weights), size=size, replace=True, p=self._weights)

        values = []
        for i in mixture_idx:
            gaussian_distributions = self._gaussian_distributions[i]
            values.append(gaussian_distributions.sample(1))

        return np.fromiter(values, dtype=np.float64)
