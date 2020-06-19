import numpy as np

from shfl.federated_aggregator.federated_aggregator import FederatedAggregator


class WeightedFedAvgAggregator(FederatedAggregator):
    """
    Implementation of Weighted Federated Avegaring Aggregator. The aggregation of the parameters is based in the number of data \
    in every node.

    It implements [Federated Aggregator](../Federated Aggregator)
    """

    def aggregate_weights(self, clients_params):
        """
        Implementation of abstract method of class [AggregateWeightsFunction](../Federate Aggregator/#federatedaggregator-class)
        # Arguments:
            clients_params: list of multi-dimensional (numeric) arrays. Each entry in the list contains the model's parameters of one client.

        # Returns:
            aggregated_weights: aggregator weights representing the global learning model
        """
        clients_params_array = np.array(clients_params)

        num_clients = clients_params_array.shape[0]
        num_layers = clients_params_array.shape[1]

        ponderated_weights = np.array([self._percentage[client] * clients_params_array[client, :] for client in range(num_clients)])
        aggregated_weights = np.array([np.sum(ponderated_weights[:, layer], axis=0) for layer in range(num_layers)])

        return aggregated_weights
