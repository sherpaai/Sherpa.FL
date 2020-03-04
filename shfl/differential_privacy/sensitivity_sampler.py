import numpy as np
import scipy
from math import pow


class SensitivitySampler:
    """
    This class implements the algorithm described in the article
    Benjamin I. P. Rubinstein and Francesco Aldà. "Pain-Free Random Differential Privacy with Sensitivity Sampling",
    accepted into the 34th International Conference on Machine Learning (ICML'2017), May 2017.
    It provides a method to estimate the sensitivity of a generic query using a concrete sensitivity norm.
    """
    def sample_sensitivity(self, query, sensitivity_norm, oracle, n=None, m=None, gamma=None):
        """
        This method calculates the parameters to sample the oracle and estimates the sensitivity

        Parameters
        ----------
        query : ~Query
            Function to apply over private data
        sensitivity_norm : function
            Function to compute the sensitivity norm
        oracle : ~ProbabilityDistribution
            ProbabilityDistribution to sample.
        n: int
            Size of private data
        m: int
            Size of sampling
        gamma: float
            Privacy confidence level

        Return
        ------
        sensitivity : float
            Calculated sensitivity value by the sampler
        mean : float
            Mean sensitivity from all samples.
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
        gs = np.ones(m) * np.inf

        for i in range(0, m):
            db1 = oracle.sample(n-1)
            db2 = db1
            db1 = np.concatenate((db1, oracle.sample(1)))
            db2 = np.concatenate((db2, oracle.sample(1)))
            gs[i] = self._sensitivity_norm(query, sensitivity_norm, db1, db2)

        gs = np.sort(gs)
        return gs[k-1], np.mean(gs)

    def _sensitivity_norm(self, query, sensitivity_norm, x1, x2):
        value_1 = query.get(x1)
        value_2 = query.get(x2)

        return sensitivity_norm.compute(value_1, value_2)

    def _sensitivity_sampler_config(self, m, gamma):
        if m is None:
            lambert_value = np.real(scipy.special.lambertw(-gamma / (2 * np.exp(0.5)), 1))
            rho = np.exp(lambert_value + 0.5)
            m = np.ceil(np.log(1 / rho) / (2 * pow((gamma - rho), 2)))
            gamma_lo = rho + np.sqrt(np.log(1 / rho) / (2 * m))
            k = np.ceil(m * (1 - gamma + gamma_lo))
        else:
            rho = np.exp(np.real(scipy.special.lambertw(-1 / (4 * m), 1)) / 2)
            gamma_lo = rho + np.sqrt(np.log(1 / rho) / (2 * m))
            if gamma is None:
                gamma = gamma_lo
                k = m
            else:
                k = np.ceil(m * (1 - gamma + gamma_lo))

        return {'m': m, 'gamma': gamma, 'k': k, 'rho': rho}

