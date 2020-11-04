import numpy as np
import random
import tensorflow as tf

from shfl.data_base.data_base import shuffle_rows
from shfl.data_distribution.data_distribution_sampling import SamplingDataDistribution


class NonIidDataDistribution(SamplingDataDistribution):
    """
    Implementation of a non-independent and identically distributed data distribution using \
        [Data Distribution](../data_distribution/#datadistribution-class)

    In this data distribution we simulate the scenario in which clients have non-identical distribution since
    they know partially the total classes of the problem.

    This distribution only works with classification problems.
    """

    @staticmethod
    def choose_labels(num_nodes, total_labels):
        """
        Method that randomly choose labels used for each client in non-iid scenario.

        # Arguments:
            num_nodes: Number of nodes
            total_labels: Number of labels

        # Returns:
            labels_to_use
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

        return random_labels

    def make_data_federated(self, data, labels, percent, num_nodes=1, weights=None, sampling="with_replacement"):
        """
        Method that makes data and labels argument federated in a non-iid scenario.

        # Arguments:
            data: Data to federate
            labels: Labels to federate
            num_nodes: Number of nodes to create
            percent: Percent of the data (between 0 and 100) to be distributed (default is 100)
            weights: Array of weights for weighted distribution (default is None)
            sampling: methodology between with or without sampling (default "without_sampling")

        # Returns:
            federated_data: A list containing the data for each client
            federated_label: A list containing the labels for each client
        """
        if weights is None:
            weights = np.full(num_nodes, 1/num_nodes)

        # Check label's format
        if labels.ndim == 1:
            one_hot = False
            labels = tf.keras.utils.to_categorical(labels)
        else:
            one_hot = True

        # Shuffle data
        data, labels = shuffle_rows(data, labels)

        # Select percent
        data = data[0:int(percent * len(data) / 100)]
        labels = labels[0:int(percent * len(labels) / 100)]

        num_data = len(data)

        # We generate random classes for each client
        total_labels = np.unique(labels.argmax(axis=-1))
        random_classes = self.choose_labels(num_nodes, len(total_labels))

        federated_data = []
        federated_label = []

        if sampling == "with_replacement":
            for i in range(0, num_nodes):
                labels_to_use = random_classes[i]

                idx = np.array([True if i in labels_to_use else False for i in labels.argmax(axis=-1)])
                data_aux = data[idx]
                labels_aux = labels[idx]

                # Shuffle data
                data_aux, labels_aux = shuffle_rows(data_aux, labels_aux)

                percent_per_client = min(int(weights[i]*num_data), len(data_aux))

                federated_data.append(np.array(data_aux[0:percent_per_client, ]))
                federated_label.append(np.array(labels_aux[0:percent_per_client, ]))

        else:
            if sum(weights) > 1:
                weights = np.array([float(i) / sum(weights) for i in weights])

            for i in range(0, num_nodes):
                labels_to_use = random_classes[i]

                idx = np.array([True if i in labels_to_use else False for i in labels.argmax(axis=-1)])
                data_aux = data[idx]
                rest_data = data[~idx]
                labels_aux = labels[idx]
                rest_labels = labels[~idx]

                data_aux, labels_aux = shuffle_rows(data_aux, labels_aux)

                percent_per_client = min(int(weights[i] * num_data), len(data_aux))

                federated_data.append(np.array(data_aux[0:percent_per_client, ]))
                rest_data = np.append(rest_data, data_aux[percent_per_client:, ], axis=0)
                federated_label.append(np.array(labels_aux[0:percent_per_client, ]))
                rest_labels = np.append(rest_labels, labels_aux[percent_per_client:, ], axis=0)

                data = rest_data
                labels = rest_labels

        if not one_hot:
            federated_label = np.array([np.argmax(node, 1) for node in federated_label])

        return federated_data, federated_label
