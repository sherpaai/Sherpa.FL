import numpy as np
import abc


class SensitivityNorm(abc.ABC):
    """
    This class defines the interface that must be implemented to compute the sensitivity norm between
    two values in a normed space.
    """

    @abc.abstractmethod
    def compute(self, x_1, x_2):
        """
        The compute method receives the result of apply a certain function over private data and
        returns the norm of the responses

        # Arguments:
            x_1: array response from a concrete query over database 1
            x_2: array response from the same query over database 2
        """


class L1SensitivityNorm(SensitivityNorm):
    """
    Implements the L1 norm of the difference between x_1 and x_2
    """
    def compute(self, x_1, x_2):
        x = x_1 - x_2
        return np.sum(np.abs(x))

class L2SensitivityNorm(SensitivityNorm):
    """
    Implements the L2 norm of the difference between x_1 and x_2
    """
    def compute(self, x_1, x_2):
        x = x_1 - x_2
        return np.sqrt(np.sum(x**2))