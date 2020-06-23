from shfl.federated_government.federated_government import FederatedGovernment
from shfl.federated_aggregator.iowa_federated_aggregator import IowaFederatedAggregator
import numpy as np


class IowaFederatedGovernment(FederatedGovernment):
    """
    Class used to represent the IOWA Federated Government which implements [FederatedGovernment](./federated_government/#federatedgovernment-class)

    # Arguments:
        model_builder: Function that return a trainable model (see: [Model](../../model))
        federated_data: Federated data to use. (see: [FederatedData](../../private/federated_operation/#federateddata-class))
        aggregator: Federated aggregator function (see: [Federated Aggregator](../../federated_aggregator))
        model_param_access: Policy to access model's parameters, by default non-protected (see: [DataAccessDefinition](../data/#dataaccessdefinition))
        dynamic: boolean indicating if we use the dynamic or static version (default True)
        a: first argument of linguistic quantifier (default 0)
        b: second argument of linguistic quantifier (default 0.2)
        c: third argument of linguistic quantifier (default 0.8)
        y_b: fourth argument of linguistic quantifier (default 0.4)
        k: distance param of the dynamic version (default 3/4)
    """

    def __init__(self, model_builder, federated_data, model_params_access=None, dynamic=True, a=0,
                 b=0.2, c=0.8, y_b=0.4, k=3/4):
        super().__init__(model_builder, federated_data, IowaFederatedAggregator(), model_params_access)

        self._a = a
        self._b = b
        self._c = c
        self._y_b = y_b
        self._k = k
        self._dynamic = dynamic

    def performance_clients(self, data_val, label_val):
        """
        Evaluation of local learning models over global test dataset.

        # Arguments:
            val_data: validation dataset
            val_label: corresponding labels to validation dataset

        # Returns:
            client_performance: Performance for each client.
        """
        client_performance = []
        for data_node in self._federated_data:
            # Predict local model in test
            local_performance = data_node.performance(data_val, label_val)
            client_performance.append(local_performance)

        return np.array(client_performance)

    def run_rounds(self, n, test_data, test_label):
        """
        Implementation of the abstract method of class [Learning Approach](../Learning Approach/#learningapproach-class)

        Run one more round beginning in the actual state testing in test data and federated_local_test.

        # Arguments:
            n: Number of rounds
            test_data: Test data for evaluation between rounds
            test_label: Test label for evaluation between rounds

        """
        randomize = np.arange(len(test_label))
        np.random.shuffle(randomize)
        test_data = test_data[randomize, ]
        test_label = test_label[randomize]

        # Split between validation and test
        validation_data = test_data[:int(0.15*len(test_label)), ]
        validation_label = test_label[:int(0.15*len(test_label))]

        test_data = test_data[int(0.15 * len(test_label)):, ]
        test_label = test_label[int(0.15 * len(test_label)):]

        for i in range(0, n):
            print("Accuracy round " + str(i))
            self.deploy_central_model()
            self.train_all_clients()
            self.evaluate_clients(test_data, test_label)
            client_performance = self.performance_clients(validation_data, validation_label)
            self._aggregator.set_ponderation(client_performance, self._dynamic, self._a, self._b, self._c, self._y_b,
                                             self._k)
            self.aggregate_weights()
            self.evaluate_global_model(test_data, test_label)
            print("\n\n")
