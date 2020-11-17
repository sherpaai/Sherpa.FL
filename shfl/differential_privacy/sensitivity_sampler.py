import numpy as np
from scipy import special
from math import pow
from multipledispatch import dispatch
from multipledispatch.variadic import Variadic
import copy


class SensitivitySampler:
    """
    This class implements the algorithm described in the article
    Benjamin I. P. Rubinstein and Francesco Aldà "Pain-Free Random Differential Privacy with Sensitivity Sampling",
    accepted into the 34th International Conference on Machine Learning (ICML'2017), May 2017.
    It provides a method to estimate the sensitivity of a generic query using a concrete sensitivity norm.

    # References
        - [Pain-Free Random Differential Privacy with Sensitivity Sampling](
           https://arxiv.org/pdf/1706.02562.pdf)
    """

    def sample_sensitivity(self, query, sensitivity_norm, oracle, n, m=None, gamma=None):
        """
        This method calculates the parameters to sample the oracle and estimates the sensitivity.
        One of m or gamma must be provided.

        # Arguments:
            query: Function to apply over private data (see: [Query](../../private/query))
            sensitivity_norm: Function to compute the sensitivity norm
                (see: [Norm](../norm))
            oracle: ProbabilityDistribution to sample.
            n: int for size of private data
            m: int for size of sampling
            gamma: float for privacy confidence level

        # Returns:
            sensitivity: Calculated sensitivity value by the sampler
            mean: Mean sensitivity from all samples.
        """
        sensitivity_sampler_config = self._sensitivity_sampler_config(m=m, gamma=gamma)

        sensitivity, mean = self._sensitivity_sampler(
            query=query,
            sensitivity_norm=sensitivity_norm,
            oracle=oracle,
            n=n,
            m=int(sensitivity_sampler_config['m']),
            k=int(sensitivity_sampler_config['k']))
        return sensitivity, mean

    def _sensitivity_sampler(self, query, sensitivity_norm, oracle, n, m, k):
        """
        It samples the sensitivity by applying the algorithm described in
        [Pain-Free Random Differential Privacy with Sensitivity Sampling](https://arxiv.org/pdf/1706.02562.pdf)

        # Arguments:
            query: Function to apply over private data (see: [Query](../../private/query))
            sensitivity_norm: Function to compute the sensitivity norm
                (see: [Norm](../norm))
            oracle: ProbabilityDistribution to sample.
            n: int for size of private data
            m: int for number of sensitivity samples
            k: element which contains the highest sampled value

        # Returns:
            a tuple with the sampled sensitivity and the mean of the sampled sensitivities
        """
        gs = [np.inf for i in range(m)]
        
        for i in range(0, m):
            db1 = oracle.sample(n - 1)
            db2 = db1
            db1 = self._concatenate(db1, oracle.sample(1))
            db2 = self._concatenate(db2, oracle.sample(1))
            gs[i] = self._sensitivity_norm(query, sensitivity_norm, db1, db2)
            
        return self._sort_sensitivity(*gs, k=k)
    
    @staticmethod
    def _seq_iter(obj):
        return obj if isinstance(obj, dict) else range(len(obj))
    
    @dispatch((np.ScalarType, np.ndarray), (np.ScalarType, np.ndarray))
    def _concatenate(self, x_1, x_2):
        return np.concatenate((x_1, x_2))
    
    @dispatch((list, dict), (list, dict))
    def _concatenate(self, x_1, x_2):
        output = copy.deepcopy(x_1)
        for i, j in zip(self._seq_iter(x_1), self._seq_iter(x_2)):
            output[i] = self._concatenate(x_1[i], x_2[j])
        return output

    @dispatch(Variadic[(np.ndarray, list)])
    def _sort_sensitivity(self, *gs, k):
        """
        Sort sensitivity. Items to sort are iterables:
        either ndarrays or lists of ndarrays.
        """
        gs_sorted = [np.sort(np.array(item), axis=0) for item in zip(*gs)]
        gs_max = [item[k - 1] for item in gs_sorted]
        gs_mean = [np.mean(item, axis=0) for item in gs_sorted]

        return gs_max, gs_mean

    @dispatch(Variadic[np.ScalarType])
    def _sort_sensitivity(self, *gs, k):
        """
        Sort sensitivity. Items to sort are scalars.
        """
        gs = [[item] for item in gs]
        [gs_max], [gs_mean] = self._sort_sensitivity(*gs, k=k)

        return gs_max, gs_mean

    @staticmethod
    def _sensitivity_norm(query, sensitivity_norm, x1, x2):
        """
        This method queries databases x1 and x2 and computes the norm
        of the difference of the results.

        # Arguments:
            query: Function to apply over private data (see: [Query](../../private/query))
            sensitivity_norm: Function to compute the sensitivity norm
                (see: [Norm](../norm))
            x1: database to be queried
            x2: database to be queried

        # Returns:
            The norm of the difference of the queries.
        """
        value_1 = query.get(x1)
        value_2 = query.get(x2)

        return sensitivity_norm.compute(value_1, value_2)

    @staticmethod
    def _sensitivity_sampler_config(m, gamma):
        """
        This method computes the optimal values for m, gamma, k and rho

        # Arguments:
            m: int for size of sampling
            gamma: float for privacy confidence level

        # Returns:
            A dictionary with the computed values
        """
        if m is None:
            lambert_value = np.real(special.lambertw(-gamma / (2 * np.exp(0.5)), 1))
            rho = np.exp(lambert_value + 0.5)
            m = np.ceil(np.log(1 / rho) / (2 * pow((gamma - rho), 2)))
            gamma_lo = rho + np.sqrt(np.log(1 / rho) / (2 * m))
            k = np.ceil(m * (1 - gamma + gamma_lo))
        else:
            rho = np.exp(np.real(special.lambertw(-1 / (4 * m), 1)) / 2)
            gamma_lo = rho + np.sqrt(np.log(1 / rho) / (2 * m))
            if gamma is None:
                gamma = gamma_lo
                k = m
            else:
                k = np.ceil(m * (1 - gamma + gamma_lo))

        return {'m': m, 'gamma': gamma, 'k': k, 'rho': rho}
    
    