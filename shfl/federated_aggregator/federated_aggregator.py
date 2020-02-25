import numpy as np
import abc


class FederatedAggregator(abc.ABC):
    """
    Interface for Federated Aggregator

    Attributes
    ---------
    percentage : float
        Percentage of total data for each client
    accuracy_models : array
        Accuracy for each model
    """

    def __init__(self, percentage=None, accuracy_models=None):
        self._percentage = percentage
        self._accuracy_models = np.array(accuracy_models)

    @abc.abstractmethod
    def aggregate_weights(self, clients_params):
        """
        Abstract method that aggregates the weights of the client models.
        """
