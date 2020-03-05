import numpy as np

from shfl.private.query import Mean
from shfl.differential_privacy.probability_distribution import NormalDistribution
from shfl.differential_privacy import SensitivitySampler
from shfl.differential_privacy import L1SensitivityNorm


def test_sample_sensitivity_gamma():
    distribution = NormalDistribution(0, 1)

    sampler = SensitivitySampler()
    sensitivity, mean = sampler.sample_sensitivity(Mean(), L1SensitivityNorm(), distribution, n=100, gamma=0.33)

    assert np.abs(mean - 0) < 0.5


def test_sample_sensitivity_m():
    distribution = NormalDistribution(0, 1)

    sampler = SensitivitySampler()
    sensitivity, mean = sampler.sample_sensitivity(Mean(), L1SensitivityNorm(), distribution, n=100, m=285)

    assert np.abs(mean - 0) < 0.5


def test_sample_sensitivity_gamma_m():
    distribution = NormalDistribution(0, 1)

    sampler = SensitivitySampler()
    sensitivity, mean = sampler.sample_sensitivity(Mean(), L1SensitivityNorm(), distribution, n=100, m=285, gamma=0.33)

    assert np.abs(mean - 0) < 0.5