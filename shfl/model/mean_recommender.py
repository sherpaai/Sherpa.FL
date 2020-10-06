import numpy as np

from shfl.model.recommender import Recommender


class MeanRecommender(Recommender):
    """
    Implementation of a simple recommender using [Recommender](../model/#recommender-class).

    For each client, given a set of labels in the training set, the recommender computes the mean value
    and uses it to make predictions.
    """

    def __init__(self):
        super().__init__()
        self._mu = None

    def train_recommender(self, data, labels):
        """
        Method that trains the model

        # Arguments:
            data: Data to train the model
            labels: Label for each train element
        """
        self._mu = np.mean(labels)

    def predict_recommender(self, data):
        """
        Predict labels for data

        # Arguments:
            data: Data for predictions. Only includes the data of this client

        # Returns:
            predictions: Array with predictions for data
        """
        predictions = np.full(len(data), self._mu)
        return predictions

    def evaluate_recommender(self, data, labels):
        """
        This method must returns the root mean square error

        # Arguments:
            data: Data to be evaluated. Only includes the data of this client
            labels: True values of data of this client
        """
        predictions = self.predict(data)
        if predictions.size == 0:
            rmse = 0
        else:
            rmse = np.sqrt(np.mean((predictions - labels) ** 2))
        return rmse

    def get_model_params(self):
        """
        Gets the params that define the model

        # Returns:
            params: Mean rating
        """
        return self._mu

    def set_model_params(self, params):
        """
        Update the params that define the model

        # Arguments:
            params: Parameter defining the model
        """
        self._mu = params

    def performance_recommender(self, data, labels):
        """
        This method returns the root mean square error of the recommender.

        # Arguments:
            data: Data to be evaluated. Only includes the data of this client
            labels: True values of data of this client
        """
        predictions = self.predict(data)
        if predictions.size == 0:
            rmse = 0
        else:
            rmse = np.sqrt(np.mean((predictions - labels) ** 2))
        return rmse
