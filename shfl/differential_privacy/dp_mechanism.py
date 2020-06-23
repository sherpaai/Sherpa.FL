import numpy as np
import scipy
from math import sqrt
from math import log

from shfl.private.data import DPDataAccessDefinition
from shfl.private.query import IdentityFunction


class RandomizedResponseCoins(DPDataAccessDefinition):
    """
    This class uses a simple mechanism to add randomness for binary data. Both the input and output are binary
    arrays or scalars. This algorithm is described by Cynthia Dwork and Aaron Roth in "The algorithmic Foundations of Differential Privacy".

    1.- Flip a coin

    2.- If tails, then respond truthfully.

    3.- If heads, then flip a second coin and respond "Yes" if heads and "No" if tails.

    Input data must be binary, otherwise exception will be raised.

    This method is log(3)-differentially private

    # Arguments
        prob_head_first: float in [0,1] representing probability to use a random response instead of true value.
            This is equivalent to prob_head of the first coin flip algorithm described by Dwork.
        prob_head_second: float in [0,1] representing probability of respond true when random answer is provided.
            Equivalent to prob_head in the second coin flip in the algorithm.

    # Properties:
        epsilon_delta: Return epsilon_delta value

    # References
        - [The algorithmic foundations of differential privacy](
           https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
    """

    def __init__(self, prob_head_first=0.5, prob_head_second=0.5):
        self._prob_head_first = prob_head_first
        self._prob_head_second = prob_head_second
        self._epsilon_delta = (log(3), 0)

    @property
    def epsilon_delta(self):
        return self._epsilon_delta

    def apply(self, data):
        """
        This method applies the the two coin flip algorithm described by Dwork to the given data,
        to access the data. Both the input and output of the method are binary arrays.

        # Arguments:
            data: data to be accessed. It can be either a scalar or a numpy array made of scalars.

        # Returns:
            Queried data with differential privacy.
        """
        data = np.asarray(data)
        self._check_binary_data(data)

        first_coin_flip = scipy.stats.bernoulli.rvs(
            p=(1 - self._prob_head_first), size=data.shape)
        second_coin_flip = scipy.stats.bernoulli.rvs(
            p=self._prob_head_second, size=data.shape)

        result = data * first_coin_flip + \
            (1 - first_coin_flip) * second_coin_flip

        return result


class RandomizedResponseBinary(DPDataAccessDefinition):
    """
    Implements the most general binary randomized response algorithm. Both the input and output are binary
    arrays or scalars. The algorithm is defined through the conditional probabilities

    - p00 = P( output=0 | input=0 ) = f0
    - p10 = P( output=1 | input=0 ) = 1 - f0
    - p11 = P( output=1 | input=1 ) = f1
    - p01 = P( output=0 | input=1 ) = 1 - f1

    For f0=f1=0 or 1, the algorithm is not random. It is maximally random for f0=f1=1/2.
    This class contains, for special cases of f0, f1, the class RandomizedResponseCoins.
    This algorithm is epsilon-differentially private if epsilon >= log max{ p00/p01, p11/p10} = log \
    max { f0/(1-f1), f1/(1-f0)}

    Input data must be binary, otherwise exception will be raised.

    # Arguments
        f0: float in [0,1] representing the probability of getting 0 when the input is 0
        f1: float in [0,1] representing the probability of getting 1 when the input is 1

    # Properties:
        epsilon_delta: Return epsilon_delta value

    # References
        - [Using Randomized Response for Differential PrivacyPreserving Data Collection](http://csce.uark.edu/~xintaowu/publ/DPL-2014-003.pdf)
    """

    def __init__(self, f0, f1, epsilon):
        self._check_epsilon_delta((epsilon, 0))
        if f0 <= 0 or f0 >= 1:
            raise ValueError(
                "f0 argument must be between 0 an 1, {} was provided".format(f0))
        if f1 <= 0 or f1 >= 1:
            raise ValueError(
                "f1 argument must be between 0 an 1, {} was provided".format(f1))
        if epsilon < log(max(f0 / (1 - f1), f1 / (1 - f0))):
            raise ValueError("To ensure epsilon differential privacy, the following inequality mus be satisfied " +
                             "{}=epsilon >= {}=log max ( f0 / (1 - f1), f1 / (1 - f0))"
                             .format(epsilon, log(max(f0 / (1 - f1), f1 / (1 - f0)))))
        self._f0 = f0
        self._f1 = f1
        self._epsilon = epsilon

    @property
    def epsilon_delta(self):
        return self._epsilon, 0

    def apply(self, data):
        """
        This method applies the randomized response mechanism to the given data, to access the data
        Both the input and output of the method are binary arrays.

        # Arguments:
            data: data to be accessed. It can be either a scalar or a numpy array made of scalars.

        # Returns:
            Queried data with differential privacy.
        """
        data = np.asarray(data)
        self._check_binary_data(data)

        probabilities = np.empty(data.shape)
        x_zero = data == 0
        probabilities[x_zero] = 1 - self._f0
        probabilities[~x_zero] = self._f1
        x_response = scipy.stats.bernoulli.rvs(p=probabilities)

        return x_response


class LaplaceMechanism(DPDataAccessDefinition):
    """
    Implements the Laplace mechanism for differential privacy defined by Dwork in
    "The algorithmic Foundations of Differential Privacy".

    Notice that the Laplace mechanism is a randomization algorithm that depends on the l1 sensitivity,
    which can be regarded as a numeric query. One can show that this mechanism is
    epsilon-differentially private with epsilon = l1-sensitivity/b where b is a constant.

    In order to apply this mechanism for a particular value of epsilon, we need to compute
    the sensitivity, which might be hard to compute in practice. The framework provides
    a method to estimate the sensitivity of a query that maps the private data in a normed space
    (see: [SensitivitySampler](../sensitivity_sampler))

    # Arguments:
        sensitivity: float or array representing sensitivity of the applied query
        epsilon: float for the epsilon you want to apply
        query: Function to apply over private data (see: [Query](../../private/query)). This parameter is optional and \
            the identity function (see: [IdentityFunction](../../private/query/#identityfunction-class)) will be used \
            if it is not provided.

    # Properties:
        epsilon_delta: Return epsilon_delta value

    # References
        - [The algorithmic foundations of differential privacy](
           https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
    """

    def __init__(self, sensitivity, epsilon, query=IdentityFunction()):
        self._check_epsilon_delta((epsilon, 0))
        self._check_sensitivity_positive(sensitivity)

        self._sensitivity = sensitivity
        self._epsilon = epsilon
        self._query = query

    @property
    def epsilon_delta(self):
        return self._epsilon, 0

    def apply(self, data):
        """
        This method applies the laplace mechanism to the given data, to access the data

        # Arguments:
            data: data to be accessed. It can be either a scalar or a numpy array made of scalars.

        # Returns:
            Queried data with differential privacy.
        """
        query_result = np.asarray(self._query.get(data))
        sensitivity = np.asarray(self._sensitivity)
        self._check_sensitivity_shape(sensitivity, query_result)
        b = sensitivity / self._epsilon

        return query_result + np.random.laplace(loc=0.0, scale=b, size=query_result.shape)


class GaussianMechanism(DPDataAccessDefinition):
    """
    Implements the Gaussian mechanism for differential privacy defined by Dwork in
    "The algorithmic Foundations of Differential Privacy".

    Notice that the Gaussian mechanism is a randomization algorithm that depends on the l2-sensitivity,
    which can be regarded as a numeric query. One can show that this mechanism is
    (epsilon, delta)-differentially private where noise is draw from a Gauss Distribution with zero mean
    and standard deviation equal to sqrt(2 * ln(1,25/delta)) * l2-sensivity / epsilon where epsilon
    is in the interval (0, 1)

    In order to apply this mechanism, we need to compute
    the l2-sensitivity, which might be hard to compute in practice. The framework provides
    a method to estimate the sensitivity of a query that maps the private data in a normed space
    (see: [SensitivitySampler](../sensitivity_sampler))

    # Arguments:
        sensitivity: float or array representing l2-sensitivity of the applied query
        epsilon: float for the epsilon you want to apply
        delta: float for the delta you want to apply
        query: Function to apply over private data (see: [Query](../../private/query)). This parameter is optional and \
            the identity function (see: [IdentityFunction](../../private/query/#identityfunction-class)) will be used \
            if it is not provided.

    # Properties:
        epsilon_delta: Return epsilon_delta value

    # References
        - [The algorithmic foundations of differential privacy](
           https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
    """

    def __init__(self, sensitivity, epsilon_delta, query=IdentityFunction()):
        self._check_epsilon_delta(epsilon_delta)
        if epsilon_delta[0] >= 1:
            raise ValueError(
                "In the Gaussian mechanism epsilon have to be greater than 0 and less than 1")
        self._check_sensitivity_positive(sensitivity)
        self._sensitivity = sensitivity
        self._epsilon_delta = epsilon_delta
        self._query = query

    @property
    def epsilon_delta(self):
        return self._epsilon_delta

    def apply(self, data):
        """
        This method applies the gaussian mechanism to the given data, to access the data.

        # Arguments:
            data: data to be accessed. It can be either a scalar or a numpy array made of scalars

        # Returns:
            Queried data with differential privacy.
        """
        query_result = np.asarray(self._query.get(data))
        sensitivity = np.asarray(self._sensitivity)
        self._check_sensitivity_shape(sensitivity, query_result)
        std = sqrt(2 * np.log(1.25 / self._epsilon_delta[1])) * \
            sensitivity / self._epsilon_delta[0]

        return query_result + np.random.normal(loc=0.0, scale=std, size=query_result.shape)


class ExponentialMechanism(DPDataAccessDefinition):
    """
    Implements the exponential mechanism differential privacy defined by Dwork in
    "The algorithmic Foundations of Differential Privacy".

    # Arguments:
        u: utility function with arguments x and r. It should be vectorized, so that for a \
        particular database x, it returns as many values as given in r.
        r: array for the response space.
        delta_u: float for the sensitivity of the utility function.
        epsilon: float for the epsilon you want to apply.
        size: integer for the number of queries to perform at once. If not given it defaults to one.

    # References
        - [The algorithmic foundations of differential privacy](
           https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
    """

    def __init__(self, u, r, delta_u, epsilon, size=1):
        self._check_epsilon_delta((epsilon, 0))
        self._u = u
        self._r = r
        self._delta_u = delta_u
        self._epsilon = epsilon
        self._size = size

    @property
    def epsilon_delta(self):
        return self._epsilon, 0

    def apply(self, data):
        """
        This method applies the exponetial mechanism to the given data, to access the data.

        # Arguments:
            data: data to be accessed. It can be either a scalar or a numpy array made of scalars

        # Returns:
            Queried data with differential privacy.
        """
        r_range = self._r
        u_points = self._u(data, r_range)
        p = np.exp(self._epsilon * u_points / (2 * self._delta_u))
        p /= p.sum()
        sample = np.random.choice(
            a=r_range, size=self._size, replace=True, p=p)

        return sample
