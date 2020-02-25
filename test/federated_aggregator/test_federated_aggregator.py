import numpy as np
import pytest

from shfl.federated_aggregator.federated_aggregator import FederatedAggregator


class TestFederatedAggregator(FederatedAggregator):
    def aggregate_weights(self, clients_params):
        pass


def test_federated_aggregator_private_data():
    percentage = 100
    acc_models = np.random.rand(10)
    fa = TestFederatedAggregator(percentage, acc_models)

    assert fa._percentage == percentage
    assert np.array_equal(acc_models,fa._accuracy_models)
