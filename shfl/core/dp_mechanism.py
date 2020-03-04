import numpy as np
import abc


class DifferentialPrivacyMechanism(abc.ABC):
    """
    This is the interface that must be implemented for every differential privacy method
    """
    @abc.abstractmethod
    def randomize(self, data):
        """
        The method should add some noise to the original data and return the obtained value
        """


class UnrandomizedMechanism(DifferentialPrivacyMechanism):
    """
    This class doesn't implement randomization mechanism
    """
    def randomize(self, data):
        """
        Data is returned without modification and no differential privacy is applied
        """
        return data


class RandomizeBinaryProperty(DifferentialPrivacyMechanism):
    """
    This class uses simple mechanism to add randomness for binary data. This algorithm is described
    by Dwork in her work "The algorithmic Foundations of Differential Privacy".
    1.- Flip a coin
    2.- If tails, then respond truthfully.
    3.- If heads, then flip a second coin and respond "Yes" if heads and "No" if tails.

    Attributes
    ----------
    _prob_head_first : float
        Probability to use a random response instead of true value. This is equivalent to prob_head of the first coin
        flip algorithm described by Dwork.
    _prob_head_second : float
        Probability of respond true when random answer is provided. Equivalent to prob_head in the second coin flip
        in the algorithm.
    """
    def __init__(self, prob_head_first=0.5, prob_head_second=0.5):
        self._prob_head_first = prob_head_first
        self._prob_head_second = prob_head_second

    def randomize(self, data):
        """
        Implements the two coin flip algorithm described by Dwork.
        """
        if data != 0 and data != 1:
            raise ValueError("RandomizeBinaryProperty works with binary data, but input is not binary")

        random_value = np.random.rand()
        if random_value > self._prob_head_first:
            return int(data)

        random_value = np.random.rand()
        if random_value > self._prob_head_second:
            return 0

        return 1


class LaplaceMechanism(DifferentialPrivacyMechanism):
    """
    Implements the Laplace Mechanism for differential privacy
    """
    def __init__(self, sensitivity, epsilon):
        self._sensitivity = sensitivity
        self._epsilon = epsilon

    def randomize(self, data):
        b = self._sensitivity/self._epsilon
        return data + np.random.laplace(loc=0.0, scale=b, size=len(data))
