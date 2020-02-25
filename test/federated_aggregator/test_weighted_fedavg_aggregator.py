import numpy as np
import pytest

from shfl.federated_aggregator.weighted_avgfed_aggregator import WeightedAvgFedAggregator

def test_aggregated_weights():
    num_clients = 10
    num_layers = 5
    tams = [[128,64],[64,64],[64,64],[64,32],[32,10]]

    weights = []
    for i in range(num_clients):
        weights.append([np.random.rand(tams[j][0],tams[j][1]) for j in range(num_layers)])

    clients_params = np.array(weights)

    percentage = np.random.dirichlet(np.ones(num_clients),size=1)[0]

    avgfa = WeightedAvgFedAggregator(percentage=percentage)
    aggregated_weights = avgfa.aggregate_weights(clients_params)

    own_ponderated_weights = np.array([percentage[client] * clients_params[client, :] for client in range(num_clients)])
    own_agg = np.array([np.sum(own_ponderated_weights[:, layer], axis=0) for layer in range(num_layers)])

    for i in range(num_layers):
        assert np.array_equal(own_agg[i],aggregated_weights[i])
    assert aggregated_weights.shape[0] == num_layers

