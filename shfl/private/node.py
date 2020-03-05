import copy

from shfl.private.data import UnprotectedAccess


class DataNode:
    """
    This class represents an independent data node
    """

    def __init__(self):
        self._private_data = {}
        self._private_data_access_policies = {}
        self._model = None
        self._model_access_policy = UnprotectedAccess()

    @property
    def private_data(self):
        """
        Allows to see data for this node, but you cannot retrieve data

        Returns
        ------
        private : object
            test data
        """
        print("Node private data, you can see the data for debug purposes but the data remains in the node")
        print(type(self._private_data))
        print(self._private_data)

    def set_private_data(self, name, data):
        self._private_data[name] = copy.deepcopy(data)

    def configure_private_data_access(self, name, data_access_definition):
        self._private_data_access_policies[name] = copy.deepcopy(data_access_definition)

    def configure_model_params_access(self, data_access_definition):
        self._model_access_policy = copy.deepcopy(data_access_definition)

    def apply_data_transformation(self, private_property, federated_transformation):
        federated_transformation.apply(self._private_data[private_property])

    def query_private_data(self, private_property):
        if private_property not in self._private_data_access_policies:
            raise ValueError("Data access must be configured before query data")

        data_access_policy = self._private_data_access_policies[private_property]
        data = data_access_policy.query.get(self._private_data[private_property])
        return data_access_policy.dp_mechanism.randomize(data)

    def query_model_params(self):
        return self._model_access_policy.query.get(self._model.get_model_params())

    @property
    def model(self):
        print("You can't get the model, you need to query the params to access")
        print(type(self._model))
        print(self._model)

    @model.setter
    def model(self, model):
        """
        Sets the model to use in the node

        Parameters
        ----------
        model: ~TrainableModel
            Instance of a class implementing ~TrainableModel
        """
        self._model = model

    def set_model_params(self, model_params):
        """
        Sets the model to use in the node

        Parameters
        ----------
        model_params: object
            Parameters to set in the model
        """
        self._model.set_model_params(model_params)

    def train_model(self, training_data_key):
        """
        Train the model that has been previously set in the data node

        Parameters
        ----------
        training_data_key: str
            String identifying the private data to use for this model
        """
        labeled_data = self._private_data.get(training_data_key)
        if not hasattr(labeled_data, 'data') or not hasattr(labeled_data, 'label'):
            raise ValueError("Private data needs to have 'data' and 'label' to train a model")
        self._model.train(labeled_data.data, labeled_data.label)

    def predict(self, data):
        return self._model.predict(data)
