from typing import Any, Union

import numpy as np
import scipy
import abc


class DifferentialPrivacyMechanism(abc.ABC):
    """
    This is the interface that must be implemented to create an algorithm with the goal to protect
    information
    """
    @abc.abstractmethod
    def randomize(self, data):
        """
        The method should add some noise to the original data and return the obtained value
        """


class UnrandomizedMechanism(DifferentialPrivacyMechanism):
    """
    This class doesn't implement randomization mechanism. You might want to send the data without applying
    any differential privacy method. Maybe your algorithm is private by design and it is not important that
    someone intercepts your data.
    """
    def randomize(self, data):
        """
        Data is returned without modification and no differential privacy is applied
        """
        return data


class RandomizeBinaryProperty(DifferentialPrivacyMechanism):
    """
    This class uses simple mechanism to add randomness for binary data. This algorithm is described
    by Cynthia Dwork and Aaron Roth in their work "The algorithmic Foundations of Differential Privacy".

    1.- Flip a coin

    2.- If tails, then respond truthfully.

    3.- If heads, then flip a second coin and respond "Yes" if heads and "No" if tails.

    # Arguments
        prob_head_first: float in [0,1] representing probability to use a random response instead of true value.
            This is equivalent to prob_head of the first coin flip algorithm described by Dwork.
        prob_head_second: float in [0,1] representing probability of respond true when random answer is provided.
            Equivalent to prob_head in the second coin flip in the algorithm.

    # References
        - [The algorithmic foundations of differential privacy](
           https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
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


class RandomizedResponseBinary(DifferentialPrivacyMechanism):
    """
    P(1|1) = f1
    P(0|0) = f2

    For f1=f2=0 or 1, the algorithm is not random. It is maximally random for f1=f2=1/2.
    This class contains, for special cases of f1, f2, the class RandomizeBinaryProperty.

    # Arguments
        f1: float in [0,1]
        f2: float in [0,1]
    """

    def __init__(self, f1, f2):
        self._f1 = f1
        self._f2 = f2

    def randomize(self, data):
        """
        Implements the general binary randomized response algorithm.

        Both the input and output of the method are binary arrays.
        """
        x_response = np.zeros(len(data))
        x_zero = data == 0
        x_response[x_zero] = scipy.stats.bernoulli.rvs(1 - self._f2, size=sum(x_zero))
        x_response[~x_zero] = scipy.stats.bernoulli.rvs(self._f1, size=len(data)-sum(x_zero))

        return x_response

class LaplaceMechanism(DifferentialPrivacyMechanism):
    """
    Implements the laplace mechanism for differential privacy defined by Dwork in their work
    "The algorithmic Foundations of Differential Privacy".

    Notice that the Laplace mechanism is a randomization algorithm that depends on the sensitivity,
    which can be regarded as a numeric query. One can show that this mechanism is
    epsilon-differentially private with epsilon = sensitivity/b where b is a constant.

    In order to apply this mechanism for a particular value of epsilon, we need to compute
    the sensitivity, which might be hard to compute in practice. The framework provides
    a method to estimate the sensitivity of a query that maps the private data in a normed space
    (see: [SensitivitySampler](../Sensitivity Sampler))

    # Arguments:
        sensitivity: float representing sensitivity of the applied query
        epsilon: float for the epsilon you want to apply

    # References
        - [The algorithmic foundations of differential privacy](
           https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
    """
    def __init__(self, sensitivity, epsilon):
        self._sensitivity = sensitivity
        self._epsilon = epsilon

    def randomize(self, data):
        b = self._sensitivity/self._epsilon
        return data + np.random.laplace(loc=0.0, scale=b, size=len(data))
