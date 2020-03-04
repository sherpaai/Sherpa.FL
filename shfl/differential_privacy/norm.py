import numpy as np
import abc


class SensitivityNorm(abc.ABC):
    """
    This class computes the sensitivity norm between two values in a normed space
    """

    @abc.abstractmethod
    def compute(self, x_1, x_2):
        """
        Returns the norm of the responses

        Parameters
        ----------
        x_1: array
            Response from a concrete query over database 1
        x_2: array
            Response from the same query over database 2
        """


class L1SensitivityNorm(SensitivityNorm):
    """
    Implements the L1 norm
    """
    def compute(self, x_1, x_2):
        x = x_1 - x_2
        return np.sum(np.abs(x))
