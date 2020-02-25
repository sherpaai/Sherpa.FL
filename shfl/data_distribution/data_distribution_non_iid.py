import numpy as np
import random

from shfl.data_distribution.data_distribution import DataDistribution


def choose_labels(num_nodes, total_labels):
    """
    Method that randomly choose labels used for each client in non-iid scenario.

    Parameters
    ----------
    num_nodes : int
        Number of nodes
    total_labels : int
        Number of labels

    Return
    ------
    labels_to_use : array
        labels for each client
    """

    random_labels = []

    for i in range(0, num_nodes):
        num_labels = random.randint(2, total_labels)
        labels_to_use = []

        for j in range(num_labels):
            label = random.randint(0, total_labels - 1)
            if label not in labels_to_use:
                labels_to_use.append(label)
            else:
                while label in labels_to_use:
                    label = random.randint(0, total_labels - 1)
                labels_to_use.append(label)

        random_labels.append(labels_to_use)

    print(random_labels)
    return random_labels


class NonIidDataDistribution(DataDistribution):
    """
    Implementation of a non-independent and identically distributed data distribution

    Attributes
    ----------
    _database:
        Database to distribute
    """

    def make_data_federated(self, data, labels, num_nodes, percent, weights):
        """
        Method that makes data and labels argument federated in a non-iid scenario.

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

        weights = np.array([float(i) / sum(weights) for i in weights])

        federated_data = []
        federated_label = []

        # We generate random classes for each client
        total_labels = np.unique(labels)
        random_classes = choose_labels(num_nodes, len(total_labels))
        for i in range(0, num_nodes):
            labels_to_use = random_classes[i]

            idx = np.array([True if i in labels_to_use else False for i in labels])
            labels_aux = labels[idx]
            data_aux = data[idx, ]

            randomize = np.arange(len(labels_aux))
            np.random.shuffle(randomize)
            data_aux = data_aux[randomize, ]
            labels_aux = labels_aux[randomize]

            # Select percent
            data_aux = data_aux[0:int(percent * len(data_aux) / 100), ]
            labels_aux = labels_aux[0:int(percent * len(labels_aux) / 100)]

            percent_per_client = int(weights[i] * len(data_aux))

            federated_data.append(np.array(data_aux[0:percent_per_client, ]))
            federated_label.append(np.array(labels_aux[0:percent_per_client, ]))

        federated_data = np.array(federated_data)
        federated_label = np.array(federated_label)

        return federated_data, federated_label
