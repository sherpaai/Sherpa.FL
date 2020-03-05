import numpy as np

from shfl.private_data.probability_distribution import NormalDistribution
from shfl.private_data.probability_distribution import GaussianMixture


def test_normal_distribution():
    data_size = 1000
    array = NormalDistribution(175, 7).sample(data_size)

    assert len(array) == 1000
    assert np.abs(np.mean(array) - 175) < 5


def test_gaussian_mixture():
    data_size = 1000

    mu_M = 178
    mu_F = 162
    sigma_M = 7
    sigma_F = 7
    norm_params = np.array([[mu_M, sigma_M],
                            [mu_F, sigma_F]])

    weights = np.ones(2) / 2.0
    array = GaussianMixture(norm_params, weights).sample(data_size)

    assert len(array) == 1000
    assert np.abs(np.mean(array) - 170) < 5
