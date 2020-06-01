import abc


class LearningApproach(abc.ABC):
    """
    Abstract class Class used to represent a Learning Approach.

    # Arguments:
        model_builder: Function that return a trainable model (see: [Model](../../model))
        federated_data: Federated data to use. (see: [FederatedData](../../private/federated_operation/#federateddata-class))
        aggregator: Federated aggregator function (see: [Federated Aggregator](../../federated_aggregator))
        model_param_access: Policy to access model's parameters, by default non-protected (see: [DataAccessDefinition](../data/#dataaccessdefinition)) 
    """
    def __init__(self, model_builder, federated_data, aggregator, model_params_access=None):
        self._federated_data = federated_data
        self._model = model_builder()
        self._aggregator = aggregator
        for data_node in federated_data:
            data_node.model = model_builder()
            if model_params_access is not None:
                data_node.configure_model_params_access(model_params_access)

    @property
    def global_model(self):
        return self._model

    @abc.abstractmethod
    def train_all_clients(self):
        """
        Initialize the models of each client and train them
        """

    @abc.abstractmethod
    def aggregate_weights(self):
        """
        Calculate aggregated weights and update clients and server models
        """

    @abc.abstractmethod
    def run_rounds(self, n, test_data, test_label):
        """
        Run one more round begining in the actual state

        # Arguments:
            n: number of rounds
            test_data: test dataset
            test_label: corresponding labels to test dataset
        """
