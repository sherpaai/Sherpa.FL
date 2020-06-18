from shfl.federated_government.federated_government import FederatedGovernment
from shfl.data_distribution.data_distribution_iid import IidDataDistribution
from shfl.data_distribution.data_distribution_non_iid import NonIidDataDistribution
from shfl.federated_aggregator.cluster_fedavg_aggregator import ClusterFedAvgAggregator
from shfl.model.kmeans_model import KMeansModel
from shfl.data_base.iris import Iris

from enum import Enum
import numpy as np


class ClusteringDataBases(Enum):
    """
    Enumeration of possible databases for clustering.
    """
    IRIS = Iris


class FederatedClustering(FederatedGovernment):
    """
    Class used to represent a high-level federated clustering using k-means
    (see: [FederatedGoverment](../federated_goverment/#federatedgoverment-class)).

    # Attributes:
        * **_num_clusters, _num_features, _test_data, _test_labels**

    # Arguments:
        data_base_name_key: key of the enumeration of valid data bases (see: [ClusteringDataBases](./#clusteringdatabases-class))
        iid: boolean which specifies if the distribution if IID (True) or non-IID (False) (True by default)
        num_nodes: number of clients.
        percent: percentage of the database to distribute among nodes.
    """

    def __init__(self, data_base_name_key, iid=True, num_nodes=20, percent=100):
        if data_base_name_key in ClusteringDataBases.__members__.keys():
            module = ClusteringDataBases.__members__[data_base_name_key].value
            data_base = module()
            train_data, train_labels, test_data, test_labels = data_base.load_data()

            self._num_clusters = len(np.unique(train_labels))
            self._num_features = train_data.shape[1]

            if iid:
                distribution = IidDataDistribution(data_base)
            else:
                distribution = NonIidDataDistribution(data_base)

            federated_data, self._test_data, self._test_labels = distribution.get_federated_data(num_nodes=num_nodes,
                                                                                                 percent=percent)

            aggregator = ClusterFedAvgAggregator()

            super().__init__(self.model_builder, federated_data, aggregator)

        else:
            print("The data base name is not included. Try with: " + str(", ".join([e.name for e in ClusteringDataBases])))
            self._test_data = None

    def run_rounds(self, n=5):
        """
        Overriding of the method of run_rounds of [FederatedGoverment](../federated_goverment/#federatedgoverment-class)).

        Run one more round beginning in the actual state testing in test data and federated_local_test.

        # Arguments:
            n: Number of rounds (2 by default)
        """
        if self._test_data is not None:
            for i in range(0, n):
                print("Accuracy round " + str(i))
                self.deploy_central_model()
                self.train_all_clients()
                self.evaluate_clients(self._test_data, self._test_labels)
                self.aggregate_weights()
                self.evaluate_global_model(self._test_data, self._test_labels)
                print("\n\n")
        else:
            print("Federated images classifier is not properly initialised")

    def model_builder(self):
        """
        Build a KMeans model with the class params.

        # Returns:
            model: KMeans model.
        """
        model = KMeansModel(n_clusters=self._num_clusters, n_features=self._num_features)
        return model
