import numpy as np

from shfl.data_distribution.data_distribution import DataDistribution


class IidDataDistribution(DataDistribution):
    """
    Implementation of an independent and identically distributed data distribution

    Attributes
    ----------
    _database:
        Database to distribute
    """

    def make_data_federated(self, data, labels, num_nodes, percent, weights):
        """
        Method that makes data and labels argument federated in an iid scenario.

        Parameters
        ----------
        data: array
            Array of data
        labels: array
            labels
        num_nodes : int
            Number of nodes
        percent : int
            Percent of the data (between 0 and 100) to be distributed (default is 100)
        weights: array
            Array of weights for weighted distribution (default is None)

        Return
        ------
        federated_data : matrix
            Data for each client
        federated_labels : matrix
            Labels for each client
        """
        if weights is None:
            weights = np.full(num_nodes, 1/num_nodes)

        weights = np.array([float(i)/sum(weights) for i in weights])

        # Shuffle data
        randomize = np.arange(len(labels))
        np.random.shuffle(randomize)
        data = data[randomize, ]
        labels = labels[randomize]

        # Select percent
        data = data[0:int(percent * len(data) / 100), ]
        labels = labels[0:int(percent * len(labels) / 100)]

        sum_used = 0
        percentage_used = 0

        federated_data = []
        federated_label = []

        for client in range(0, num_nodes):
            federated_data.append(np.array(data[sum_used:int((percentage_used + weights[client]) * len(data)), ]))
            federated_label.append(np.array(labels[sum_used:int((percentage_used + weights[client]) * len(labels))]))

            sum_used = int((percentage_used + weights[client]) * len(data))
            percentage_used += weights[client]

        federated_data = np.array(federated_data)
        federated_label = np.array(federated_label)

        return federated_data, federated_label
