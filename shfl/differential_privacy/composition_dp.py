from math import sqrt, log, exp

from shfl.private.data import DPDataAccessDefinition


class ExceededPrivacyBudgetError(Exception):
    """
    This Exception is expected to be used when a certain privacy budget is exceed.
    When it is used, it means that the data cannot be accessed anymore

    # Arguments:
        epsilon_delta: the privacy budget which has been surpassed
    """

    def __init__(self, **args):
        self._epsilon_delta = None
        if args:
            if "epsilon_delta" in args:
                self._epsilon_delta = args["epsilon_delta"]

    def __str__(self):
        return 'Error: Privacy Budget {} has been exceeded'.format(self._epsilon_delta)


class AdaptiveDifferentialPrivacy(DPDataAccessDefinition):
    """
    It provides Adaptive Differential Privacy through Privacy Filters

    # Arguments:
        epsilon_delta: Tuple or array of length 2 which contains the epsilon-delta privacy budget for this data
        differentially_private_mechanism: Optional. Default method that will be used to access data. If it is not set \
        it's mandatory to pass it in every query.

    # Properties:
        epsilon_delta: Return epsilon_delta value
    """

    def __init__(self, epsilon_delta, differentially_private_mechanism=None):
        self._check_epsilon_delta(epsilon_delta)

        self._epsilon_delta = epsilon_delta
        self._epsilon_delta_access_history = []
        self._private_data_epsilon_delta_access_history = []
        if differentially_private_mechanism is not None:
            _check_differentially_private_mechanism(differentially_private_mechanism)
        self._differentially_private_mechanism = differentially_private_mechanism

    @property
    def epsilon_delta(self):
        return self._epsilon_delta

    def apply(self, data, differentially_private_mechanism=None):
        """
        It applies a differentially private mechanism if the privacy budget allows it.
        If the privacy budget is suparsed and exception (ExceededPrivacyBudgetError) is thrown.

        # Arguments:
            data: input data which is going to be accessed with differential privacy
            differentially_private_mechanism: it is the provider of differential privacy

        # Returns:
            The application of the dp-mechanism to the input data, if the privacy budget is not exceeded
        """
        differentially_private_mechanism_to_apply = self._get_data_access_definition(differentially_private_mechanism)
        self._private_data_epsilon_delta_access_history.append(differentially_private_mechanism_to_apply.epsilon_delta)

        privacy_budget_exceeded = self.__basic_adaptive_comp_theorem()
        if 0 < self._epsilon_delta[1] < exp(-1):
            privacy_budget_exceeded &= self.__advanced_adaptive_comp_theorem()
        if privacy_budget_exceeded:
            self._private_data_epsilon_delta_access_history.pop()
            raise ExceededPrivacyBudgetError(epsilon_delta=self._epsilon_delta)
        else:
            return differentially_private_mechanism_to_apply.apply(data)

    def _get_data_access_definition(self, data_access_definition):
        """
        This method checks if the given data access definition is differentially private,
        if none is provided, it ensures that the default data access definition is
        differentially private.

        # Arguments:
            data_access_definition: method to be checked for Differential Privacy

        # Returns:
            The given data_access_definition or the default one given in the constructor
        """
        if data_access_definition is not None:
            _check_differentially_private_mechanism(data_access_definition)
            return data_access_definition
        if self._differentially_private_mechanism is None:
            raise ValueError("Not data access definition provided or default method established")
        return self._differentially_private_mechanism

    def __basic_adaptive_comp_theorem(self):
        """
            It checks whether the privacy budget given by epsilon_delta is surpassed.

            It implements the theorem 3.6 from Privacy Odometers and Filters: Pay-as-you-Go Composition.

            # Returns:
                It returns True if the privacy budget if surpassed, False otherwise.

            # References:
                - [Privacy Odometers and Filters: Pay-as-you-Go Composition] (https://arxiv.org/abs/1605.08294)
        """
        global_epsilon, global_delta = self._epsilon_delta
        eps_sum, delta_sum = map(sum, zip(*self._private_data_epsilon_delta_access_history))
        return eps_sum > global_epsilon or delta_sum > global_delta

    def __advanced_adaptive_comp_theorem(self):
        """
            It checks whether the privacy budget given by epsilon_delta is surpassed.

            It implements the theorem 5.1 from Privacy Odometers and Filters: Pay-as-you-Go Composition.

            # Returns:
                It returns True if the privacy budget if surpassed, False otherwise.

            # References:
                - [Privacy Odometers and Filters: Pay-as-you-Go Composition] (https://arxiv.org/abs/1605.08294)
        """
        epsilon_history, delta_history = zip(*self._private_data_epsilon_delta_access_history)
        global_epsilon, global_delta = self._epsilon_delta

        delta_sum = sum(delta_history)
        epsilon_squared_sum = sum(epsilon ** 2 for epsilon in epsilon_history)

        h = global_epsilon ** 2 / (28.04 * log(1 / global_delta))

        a = sum(eps * (exp(eps) - 1) * 0.5 for eps in epsilon_history)
        b = epsilon_squared_sum + h
        c = 2 + log(epsilon_squared_sum / h + 1)
        d = log(2 / global_delta)

        k = a + sqrt(b * c * d)

        return k > global_epsilon or delta_sum > (global_delta * 0.5)


def _check_differentially_private_mechanism(data_access_mechanism):
    """
        This method ensures that the given data access mechanism provides Differential Privacy

        # Arguments:
            data_access_mechanism: mechanism to be checked
    """

    if not hasattr(data_access_mechanism, 'epsilon_delta'):
        raise ValueError("You can't access differentially private data with a non differentially private mechanism")
