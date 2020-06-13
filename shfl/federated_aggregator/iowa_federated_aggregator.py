from shfl.federated_aggregator.weighted_fedavg_aggregator import WeightedFedAvgAggregator

import numpy as np


class IowaFederatedAggregator(WeightedFedAvgAggregator):
    """
    Class of the IOWA version of [WeightedFedAvgAggregator](./weighted_fedavg_aggregator/#weightedfedavgaggregator-class)
    """

    def __init__(self):
        super().__init__()
        self._a = 0
        self._b = 0
        self._c = 0
        self._y_b = 0
        self._k = 0
        self._performance = None
        self._dynamic = None

    def set_ponderation(self, performance, dynamic=True, a=0, b=0.2, c=0.8, y_b=0.4, k=3/4):
        """
        Method which calculate ponderation weights of each client based on the performance vector.

        # Arguments:
            performance: vector with the performance of each local client in a validation set
            dynamic: boolean indicating if we use the dynamic or static version (default True)
            a: first argument of linguistic quantifier (default 0)
            b: second argument of linguistic quantifier (default 0.2)
            c: third argument of linguistic quantifier (default 0.8)
            y_b: fourth argument of linguistic quantifier (default 0.4)
            k: distance param of the dynamic version (default 3/4)
        """
        self._a = a
        self._b = b
        self._c = c
        self._y_b = y_b
        self._k = k
        self._performance = performance
        self._dynamic = dynamic

        self._percentage = self.get_ponderation_weights()

    def q_function(self, x):
        """
        Method that returns ponderation weights for OWA operator

        # Arguments:
            x: value of the ordering function u (orderer performance of each local model)

        # Returns
            ponderation_weights: ponderation of each client.
        """
        if x <= self._a:
            return 0
        elif x <= self._b:
            return (x - self._a) / (self._b - self._a) * self._y_b
        elif x <= self._c:
            return (x - self._b) / (self._c - self._b) * (1 - self._y_b) + self._y_b
        else:
            return 1

    def get_ponderation_weights(self):
        """
        Method that returns the value of the linguistic quantifier (Q_function) for each value x

        # Returns
            ponderation_weights: ponderation of each client.
        """

        ordered_idx = np.argsort(-self._performance)
        self._performance = self._performance[ordered_idx]
        num_clients = len(self._performance)

        ponderation_weights = np.zeros(num_clients)

        if self._dynamic:
            max_distance = self._performance[0] - self._performance[-1]
            vector_distances = np.array( [self._performance[0] - self._performance[i] for i in range(num_clients)])

            is_outlier = np.array([vector_distances[i] > self._k * max_distance for i in range(num_clients)])
            num_outliers = len(is_outlier[is_outlier is True])

            self._c = 1 - num_outliers / num_clients
            self._b = self._b * self._c

        for i in range(num_clients):
            ponderation_weights[i] = self.q_function((i + 1) / num_clients) - self.q_function(i / num_clients)

        return ponderation_weights
