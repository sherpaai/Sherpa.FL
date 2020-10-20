class FederatedGovernment:
    """
    Class used to represent the central class FederatedGoverment.

    # Arguments:
       model_builder: Function that return a trainable model (see: [Model](../model))
       federated_data: Federated data to use. (see: [FederatedData](../private/federated_operation/#federateddata-class))
       aggregator: Federated aggregator function (see: [Federated Aggregator](../federated_aggregator))
       model_param_access: Policy to access model's parameters, by default non-protected (see: [DataAccessDefinition](../private/data/#dataaccessdefinition-class))

    # Properties:
        global_model: Return the global model.
    """

    def __init__(self, model_builder, federated_data, aggregator, model_params_access=None):
        self._federated_data = federated_data
        self._model = model_builder()
        self._aggregator = aggregator
        for data_node in federated_data:
            data_node.model = self._model
            if model_params_access is not None:
                data_node.configure_model_params_access(model_params_access)

    @property
    def global_model(self):
        return self._model

    def evaluate_global_model(self, data_test, label_test):
        """
        Evaluation of the performance of the global model.

        # Arguments:
            test_data: test dataset
            test_label: corresponding labels to test dataset
        """
        evaluation = self._model.evaluate(data_test, label_test)
        print("Global model test performance : " + str(evaluation))

    def deploy_central_model(self):
        """
        Deployment of the global learning model to each client (node) in the simulation.
        """
        for data_node in self._federated_data:
            data_node.set_model_params(self._model.get_model_params())

    def evaluate_clients(self, data_test, label_test):
        """
        Evaluation of local learning models over global test dataset.

        # Arguments:
            test_data: test dataset
            test_label: corresponding labels to test dataset
        """
        for data_node in self._federated_data:
            # Predict local model in test
            evaluation, local_evaluation = data_node.evaluate(data_test, label_test)
            if local_evaluation is not None:
                print("Performance client " + str(data_node) + ": Global test: " + str(evaluation)
                     + ", Local test: " + str(local_evaluation))
            else:
                print("Test performance client " + str(data_node) + ": " + str(evaluation))

    def train_all_clients(self):
        """
        Train all the clients
        """
        for data_node in self._federated_data:
            data_node.train_model()

    def aggregate_weights(self):
        """
        Aggregate weights from all data nodes in the server model
        """
        weights = []
        for data_node in self._federated_data:
            weights.append(data_node.query_model_params())

        aggregated_weights = self._aggregator.aggregate_weights(weights)

        # Update server weights
        self._model.set_model_params(aggregated_weights)

    def run_rounds(self, n, test_data, test_label):
        """
        Run one more round beginning in the actual state testing in test data and federated_local_test.

        # Arguments:
            n: Number of rounds
            test_data: Test data for evaluation between rounds
            test_label: Test label for evaluation between rounds

        """
        for i in range(0, n):
            print("Accuracy round " + str(i))
            self.deploy_central_model()
            self.train_all_clients()
            self.evaluate_clients(test_data, test_label)
            self.aggregate_weights()
            self.evaluate_global_model(test_data, test_label)
            print("\n\n")
