import numpy as np
import abc


class Query(abc.ABC):
    """
    This class represents a query over private data
    """

    @abc.abstractmethod
    def get(self, data):
        """

        Parameters
        ----------
        data: object
            Data to process

        Returns
        -------
        answer: object
            Result of apply query over data
        """


class IdentityFunction(Query):
    """
    Implements identity function. It returns the data without modification
    """
    def get(self, data):
        return data


class Mean(Query):
    """
    Implements mean over data array
    """
    def get(self, data):
        return np.mean(data)
