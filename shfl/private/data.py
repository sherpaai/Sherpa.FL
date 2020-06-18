import abc
import numpy as np


class LabeledData:
    """
    Class to represent labeled data

    # Attributes:
        * **_data, _label**

    # Arguments:
        data: Features representing a data sample
        label: Label for this sample
    """
    def __init__(self, data, label):
        self._data = data
        self._label = label

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label


class DataAccessDefinition(abc.ABC):
    """
    Interface that must be implemented in order to define how to access the private data.
    """

    @abc.abstractmethod
    def apply(self, data):
        """
        Every implementation needs to implement this method defining how data will be returned.

        # Arguments:
            data: Raw data that are going to be accessed

        # Returns
            result_data: Result data, function of argument data
        """


class DPDataAccessDefinition(DataAccessDefinition):
    """
    Interface that must be implemented in order to define how to access differentially private data.
    Moreover, it provides some tools to ensure a proper implementation of Differential Privacy.
    """

    @staticmethod
    def _check_epsilon_delta(epsilon_delta):
        """
        It checks if the epsilon_delta parameter correctly represents the epsilon and delta values in
        epsilon-delta Differential Privacy. If the check fails, it throws an ValueError exception
        with the appropiate message

        # Arguments:
            epsilon_delta: a tuple of values, which should be the epsilon and delta values in
                epsilon-delta Differential Privacy.

        """
        if len(epsilon_delta) != 2:
            raise ValueError("epsilon_delta parameter should be a tuple with two elements, but {} were given"
                             .format(len(epsilon_delta)))
        if epsilon_delta[0] < 0:
            raise ValueError("Epsilon have to be greater than zero")
        if epsilon_delta[1] < 0:
            raise ValueError("Delta have to be greater than 0 and less than 1")

    @staticmethod
    def _check_binary_data(data):
        """
        It checks if the given argument is made of binary elements or not.
        If the check fails, it throws an ValueError exception with the appropiate message

        # Arguments:
            data: input vale which is expected to be made of binary elements.

        """
        if not np.array_equal(data, data.astype(bool)):
            raise ValueError(
                "This mechanism works with binary data, but input is not binary")

    @staticmethod
    def _check_sensitivity_positive(sensitivity):
        """
        It checks if the given sensitivy values are strictly positive (>0)

        # Arguments:
            sensitivity: sensitivity values which should be strictly positive (>0).

        If the check fails, it throws an ValueError exception with the appropiate message
        """
        sensitivity = np.asarray(sensitivity)
        if (sensitivity < 0).any():
            raise ValueError(
                "Sensitivity of the query cannot be negative")

    @staticmethod
    def _check_sensitivity_shape(sensitivity, query_result):
        """
        It checks if the given sensitivy values fit the shape of the query_result

        # Arguments:
            sensitivity: sensitivity values which should be strictly positive (>0).
            query_result: output of a query

        If the check fails, it throws an ValueError exception with the appropiate message
        """
        if sensitivity.size > 1:
            if sensitivity.size > query_result.size:
                raise ValueError(
                    "Provided more sensitivity values than query outputs")
            if not all((m == n) for m, n in zip(sensitivity.shape[::-1], query_result.shape[::-1])):
                raise ValueError("Sensitivity array dimension " + str(sensitivity.shape) +
                                 " cannot be broadcasted to query result dimension " + str(query_result.shape))

    @property
    @abc.abstractmethod
    def epsilon_delta(self):
        """
        Every differentially private mechanism needs to implement this property

        # Returns
            epsilon_delta: Privacy budget spent each time this differentially private mechanism is used

        """


class UnprotectedAccess(DataAccessDefinition):
    """
    This class implements access to data without restrictions, plain data will be returned.
    """
    def apply(self, data):
        return data
