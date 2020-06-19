from shfl.federated_aggregator.federated_aggregator import FederatedAggregator
import numpy as np
from sklearn.cluster import KMeans


class ClusterFedAvgAggregator(FederatedAggregator):
    """
    Implementation of Cluster Average Federated Aggregator.
    It adds another k-means to find the minimum distance of cluster centroids coming from each node.

    It implements [Federated Aggregator](../Federated Aggregator)
    """

    def aggregate_weights(self, clients_params):
        """
        Implementation of abstract method of class [AggregateWeightsFunction](../Federate Aggregator/#federatedaggregator-class)
        # Arguments:
            clients_params: list of multi-dimensional (numeric) arrays. Each entry in the list contains the \
             model's parameters of one client.

        # Returns:
            aggregated_weights: aggregator weights representing the global learning model
        """
        clients_params_array = np.concatenate(clients_params)

        n_clusters = clients_params[0].shape[0]
        model_aggregator = KMeans(n_clusters=n_clusters, init='k-means++')
        model_aggregator.fit(clients_params_array)
        aggregated_weights = np.array(model_aggregator.cluster_centers_)
        return aggregated_weights
