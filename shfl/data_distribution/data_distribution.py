import numpy as np
import random
import abc

from shfl.private_data.data import LabeledData
from shfl.private_data.federated_operation import FederatedData
from shfl.private_data.node import DataNode


class DataDistribution(abc.ABC):
    """
    Interface for data distribution

    Attributes
    ----------
    _database:
        Database to distribute
    """

    def __init__(self, database):
        self._database = database

    def get_federated_data(self, identifier, num_nodes, percent=100, weights=None, mistaken=0):
        """
        Method that split the whole data between the established number of nodes

        Parameters
        ----------
        identifier : str
            Name of the federated data element
        num_nodes : int
            Number of nodes to create
        percent : int
            Percent of the data (between 0 and 100) to be distributed (default is 100)
        weights: array
            Array of weights for weighted distribution (default is None)
        mistaken: int
            Number of mistaken clients

        Return
        ------
        Federated data
        """
        train_data, train_label = self._database.train
        validation_data, validation_label = self._database.validation
        test_data, test_label = self._database.test

        train_data = np.concatenate((train_data, validation_data), axis=0)
        train_label = np.concatenate((train_label, validation_label), axis=0)

        federated_train_data, federated_train_label = self.make_data_federated(train_data,
                                                                               train_label,
                                                                               num_nodes, percent,
                                                                               weights)
        if mistaken > 0:
            num_mistaken = int(mistaken * num_nodes / 100)

            mistakes = []

            for i in range(num_mistaken):
                mistaken_client = random.randint(0, num_nodes - 1)

                while mistaken_client in mistakes:
                    mistaken_client = random.randint(0, num_nodes - 1)

                mistakes.append(mistaken_client)
                random.shuffle(federated_train_label[mistaken_client])

        federated_data = FederatedData(identifier)
        for node in range(num_nodes):
            node_data = LabeledData(federated_train_data[node], federated_train_label[node])
            federated_data.add_data_node(DataNode(), node_data)

        return federated_data, test_data, test_label

    @abc.abstractmethod
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
        """
