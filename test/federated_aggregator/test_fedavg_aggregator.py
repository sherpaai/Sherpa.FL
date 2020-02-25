import numpy as np
import pytest

from shfl.federated_aggregator.avgfed_aggregator import AvgFedAggregator

def test_aggregated_weights():
    num_clients = 10
    num_layers = 5
    tams = [[128,64],[64,64],[64,64],[64,32],[32,10]]

    weights = []
    for i in range(num_clients):
        weights.append([np.random.rand(tams[j][0],tams[j][1]) for j in range(num_layers)])

    clients_params = np.array(weights)

    avgfa = AvgFedAggregator()
    aggregated_weights = avgfa.aggregate_weights(clients_params)

    own_agg = np.array([np.mean(clients_params[:, layer], axis=0) for layer in range(num_layers)])

    for i in range(num_layers):
        assert np.array_equal(own_agg[i],aggregated_weights[i])
    assert aggregated_weights.shape[0] == num_layers


