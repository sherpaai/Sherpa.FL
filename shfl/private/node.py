import copy

from shfl.private.data import UnprotectedAccess
from math import log, sqrt, exp
from warnings import warn

def basic_adaptative_comp_theorem(epsilon_delta_access_history, epsilon_delta):
    '''
        It checks wether the privacy budget given by epsilon_delta is surpassed.
        
        It implements the theorem 3.6 from Privacy Odometers and Filters: Pay-as-you-Go Composition.
        
        # Arguments:
            epsilon_delta_access_history: a list of the epsilon-delta expenses of each access
            epsilon_delta: privacy budget specified for the accessed private data
        
        # References:
            - [Privacy Odometers and Filters: Pay-as-you-Go Composition] (https://arxiv.org/abs/1605.08294)
    '''
    eps_sum, delta_sum = map(sum, zip(*epsilon_delta_access_history))
    return eps_sum > epsilon_delta[0] or delta_sum > epsilon_delta[1]


def advanced_adaptative_comp_theorem(epsilon_delta_access_history, epsilon_delta):
    '''
        It checks wether the privacy budget given by epsilon_delta is surpassed.
        
        It implements the theorem 5.1 from Privacy Odometers and Filters: Pay-as-you-Go Composition.
        
        # Arguments:
            epsilon_delta_access_history: a list of the epsilon-delta expenses of each access
            epsilon_delta: privacy budget specified for the accessed private data
        
        # References:
            - [Privacy Odometers and Filters: Pay-as-you-Go Composition] (https://arxiv.org/abs/1605.08294)
    '''
    epsilon_history, delta_history = zip(*epsilon_delta_access_history)
    global_epsilon, global_delta = epsilon_delta
    
    delta_sum = sum(delta_history)
    eps_squared_sum = sum(eps * eps for eps in epsilon_history)
    
    A = sum(eps * (eps - 1) * 0.5 for eps in epsilon_history)
    B = eps_squared_sum + global_epsilon * global_epsilon / (28.04 * log(1 / global_delta))
    C = 1 + 0.5 * log(28.04 * abs(log(1 / global_delta)) * eps_squared_sum / (global_epsilon**2) + 1)
    
    K = A + sqrt(2 * B * C * abs(log(2 / global_delta)))
    
    return K > global_epsilon or delta_sum > global_delta*0.5


class ExceededPrivacyBudgetError(Exception):
    """
    This Exception is expected to be used when a certain privacy budget is exceed. 
    When it is used, it means that the data cannot be accessed anymore

    # Arguments:
        message: this text is shown in addition to the exception text
    """
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return 'Error: Privacy Budget has been exceeded, {0} '.format(self.message)
        else:
            return 'Error: Privacy Budget has been exceeded'

class DataNode:
    """
    This class represents an independent data node.

    A DataNode has its own private data and provides methods
    to initialize this data and access to it. The access to private data needs to be configured with an access policy
    before query it or an exception will be raised. A method to transform private data is also provided. This is
    a mechanism that allows data preprocessing or related task over data. 
    
    It supports Adaptive Differential Privacy through Privacy Filters

    A model (see: [Model](../../model)) can be deployed in the DataNode and use private data
    in order to learn. It is assumed that a model is represented by its parameters and the access to these parameters
    must be also configured before queries.
    
    # Arguments:
        epsilon_delta: Tuple or array of length 2 which contains the epsilon-delta privacy budget for this data
        suppressWarning: suppress (epsilon, delta) warning which states that the basic composition theorem 
            for Adaptive Differential Privacy is going to be used
            
    """

    def __init__(self, epsilon_delta=None, suppressWarning=False):
        self._private_data = {}
        self._private_test_data = {}
        self._private_data_access_policies = {}
        self._model = None
        self._model_access_policy = UnprotectedAccess()
        self._epsilon_delta = None
        self._adaptative_pd_basic_mode = True
        if epsilon_delta is not None:
            if len(epsilon_delta) != 2:
                raise ValueError("epsilon_delta parameter has to be a tuple or list of length 2, but {} were given".format(len(epsilon_delta)))
            self._epsilon_delta = epsilon_delta
            self._epsilon_delta_access_history = []
            if self._epsilon_delta[0] <= 0:
                raise ValueError("Epsilon has to be greater than zero")
            if self._epsilon_delta[1] < 0:
                raise ValueError("Delta has to be greater than zero")
            
            if self._epsilon_delta[1] < exp(-1) and self._epsilon_delta[1] > 0:
                self._adaptative_pd_basic_mode = False
            elif not suppressWarning:
                warn("To use more efficently the (epsilon, delta) privacy budget, delta should be between 0 and exp(-1)")
    
    
    @property
    def epsilon_delta(self):
        return self._epsilon_delta
    
    @property
    def model(self):
        print("You can't get the model, you need to query the params to access")
        print(type(self._model))
        print(self._model)

    @model.setter
    def model(self, model):
        """
        Sets the model to use in the node

        # Arguments:
            model: Instance of a class implementing ~TrainableModel
        """
        self._model = model

    @property
    def private_data(self):
        """
        Allows to see data for this node, but you cannot retrieve data

        # Returns
            private : data
        """
        print("Node private data, you can see the data for debug purposes but the data remains in the node")
        print(type(self._private_data))
        print(self._private_data)

    @property
    def private_test_data(self):
        """
        Allows to see data for this node, but you cannot retrieve data

        # Returns
            private : test data
        """
        print("Node private test data, you can see the data for debug purposes but the data remains in the node")
        print(type(self._private_test_data))
        print(self._private_test_data)

    def set_private_data(self, name, data):
        """
        Creates copy of data in private memory using name as key. If there is a previous value with this key the
        data will be overridden.

        # Arguments:
            name: String with the key identifier for the data
            data: Data to be stored in the private memory of the DataNode
        """
        self._private_data[name] = copy.deepcopy(data)

    def set_private_test_data(self, name, data):
        """
        Creates copy of test data in private memory using name as key. If there is a previous value with this key the
        data will be override.

        # Arguments:
            name: String with the key identifier for the data
            data: Data to be stored in the private memory of the DataNode
        """
        self._private_test_data[name] = copy.deepcopy(data)

    def configure_data_access(self, name, data_access_definition):
        """
        Adds a DataAccessDefinition for some concrete private data.

        # Arguments:
            name: Identifier for the data that will be configured
            data_access_definition: Policy to access data (see: [DataAccessDefinition](../data/#dataaccessdefinition))
        """
        access_policy_eps_delta_available = hasattr(data_access_definition, 'epsilon_delta')
        eps_delta_available = self._epsilon_delta is not None
        
        if access_policy_eps_delta_available and not eps_delta_available:
            raise ValueError("You can't access non differentially private data with a differentially private mechanism")
        
        if not access_policy_eps_delta_available and eps_delta_available:
            raise ValueError("You can't access differentially private data with a non differentially private mechanism")
        
        self._private_data_access_policies[name] = copy.deepcopy(data_access_definition)


    def configure_model_params_access(self, data_access_definition):
        """
        Adds a DataAccessDefinition for model parameters.

        # Arguments:
            data_access_definition: Policy to access parameters (see: [DataAccessDefinition](../data/#dataaccessdefinition))
        """
        self._model_access_policy = copy.deepcopy(data_access_definition)

    def apply_data_transformation(self, private_property, federated_transformation):
        """
        Executes FederatedTransformation (see: [Federated Operation](../federated_operation)) over private data.

        # Arguments:
            private_property: Identifier for the data that will be transformed
            federated_transformation: Operation to execute (see: [Federated Operation](../federated_operation))
        """
        federated_transformation.apply(self._private_data[private_property])

    def query(self, private_property):
        """
        Queries private data previously configured. If the access didn't configured this method will raise exception

        # Arguments:
            private_property: String with the key identifier for the data
        """
        if private_property not in self._private_data_access_policies:
            raise ValueError("Data access must be configured before query data")

        data_access_policy = self._private_data_access_policies[private_property]

        access_policy_eps_delta_available = hasattr(data_access_policy, 'epsilon_delta')
        
        if access_policy_eps_delta_available:
            self._epsilon_delta_access_history.append(data_access_policy.epsilon_delta)
            
            if self._adaptative_pd_basic_mode:
                privacy_budget_exceeded = basic_adaptative_comp_theorem(self._epsilon_delta_access_history, self.epsilon_delta)
            else:
                privacy_budget_exceeded = advanced_adaptative_comp_theorem(self._epsilon_delta_access_history, self.epsilon_delta)
            if privacy_budget_exceeded:
                self._epsilon_delta_access_history.pop()
                raise ExceededPrivacyBudgetError()
            else:
                return data_access_policy.apply(self._private_data[private_property])
        else:
            return data_access_policy.apply(self._private_data[private_property])

    def query_model_params(self):
        """
        Queries model parameters. By default the parameters access is unprotected but access definition can be changed
        """
        return self._model_access_policy.apply(self._model.get_model_params())

    def set_model_params(self, model_params):
        """
        Sets the model to use in the node

        # Arguments:
            model_params: Parameters to set in the model
        """
        self._model.set_model_params(model_params)

    def train_model(self, training_data_key):
        """
        Train the model that has been previously set in the data node

        # Arguments:
            training_data_key: String identifying the private data to use for this model. This key must contain \
            LabeledData (see: [LabeledData](../data/#labeleddata))
        """
        labeled_data = self._private_data.get(training_data_key)
        if not hasattr(labeled_data, 'data') or not hasattr(labeled_data, 'label'):
            raise ValueError("Private data needs to have 'data' and 'label' to train a model")
        self._model.train(labeled_data.data, labeled_data.label)

    def predict(self, data):
        """
        Uses the model to predict new data

        # Arguments:
            data: Data to predict
        """
        return self._model.predict(data)

    def evaluate(self, data, labels):
        """
        Evaluates the performance of the model

        # Arguments:
            data: Data to predict
            labels: True values of data
        """
        return self._model.evaluate(data, labels)

    def local_evaluate(self, data_key):
        if bool(self._private_test_data):
            labeled_data = self._private_test_data.get(data_key)
            return self._model.evaluate(labeled_data.data, labeled_data.label)
        else:
            return None
