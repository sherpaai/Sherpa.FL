import copy

from shfl.private.data import UnprotectedAccess

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
    a mechanism that allows data preprocessing or related task over data. It supports Differential Privacy

    A model (see: [Model](../../model)) can be deployed in the DataNode and use private data
    in order to learn. It is assumed that a model is represented by its parameters and the access to these parameters
    must be also configured before queries.
    
    # Arguments:
        epsilon_delta: Tuple or array of length 2 which contains the epsilon-delta privacy budget for this data
    
    """

    def __init__(self, epsilon_delta=None):
        self._private_data = {}
        self._private_test_data = {}
        self._private_data_access_policies = {}
        self._model = None
        self._model_access_policy = UnprotectedAccess()
        self._epsilon_delta = None
        if epsilon_delta is not None:
            if len(epsilon_delta) != 2:
                raise ValueError("epsilon_delta parameter should be a tuple with two elements, but {} were given".format(len(epsilon_delta)))
            self._epsilon_delta = epsilon_delta
            self._epsilon_delta_access_history = []
            if self._epsilon_delta[0] <= 0:
                raise ValueError("Epsilon have to be greater than zero")
            if self._epsilon_delta[1] < 0:
                raise ValueError("Delta have to be greater than zero")

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
            eps_sum, delta_sum = map(sum, zip(*self._epsilon_delta_access_history))
            if eps_sum > self._epsilon_delta[0] or delta_sum > self._epsilon_delta[1]:
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
