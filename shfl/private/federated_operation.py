import abc
from shfl.private.node import DataNode
from shfl.private.data import LabeledData


class FederatedDataNode(DataNode):
    """
    This class represents a [DataNode](../data_node) in a FederatedData. Extends DataNode allowing
    calls to methods without explicit private data identifier, assuming access to the federated data. 
    It supports Differential Privacy

    # Arguments:
        federated_data_identifier: identifier to use in private data
        epsilon_delta:  Epsilon-delta privacy budget to be set for this data-node

    When you iterate over [FederatedData](./#federateddata-class) the kind of DataNode that you obtain is a \
    FederatedDataNode.

    # Example:

    ```python
        # Definition of federated data from dataset
        database = shfl.data_base.Emnist()
        iid_distribution = shfl.data_distribution.IidDataDistribution(database)
        federated_data, test_data, test_labels = iid_distribution.get_federated_data(num_nodes=20, percent=10)

        # Data access definition and query node 0
        federated_data.configure_data_access(UnprotectedAccess())
        federated_data[0].query()
    ```
    """
    def __init__(self, federated_data_identifier, epsilon_delta=None):
        super().__init__(epsilon_delta)
        self._federated_data_identifier = federated_data_identifier

    def query(self, private_property=None):
        if private_property is None:
            private_property = self._federated_data_identifier
        return super().query(private_property)

    def configure_data_access(self, data_access_definition):
        super().configure_data_access(self._federated_data_identifier, data_access_definition)

    def set_private_data(self, data):
        super().set_private_data(self._federated_data_identifier, data)

    def set_private_test_data(self, data):
        super().set_private_test_data(self._federated_data_identifier, data)

    def train_model(self):
        super().train_model(self._federated_data_identifier)

    def apply_data_transformation(self, federated_transformation):
        super().apply_data_transformation(self._federated_data_identifier, federated_transformation)

    def evaluate(self, data, test):
        return super().evaluate(data, test), super().local_evaluate(self._federated_data_identifier)

    def split_train_test(self, test_split=0.2):
        """
        Splits private_data in train and test sets

        # Arguments:
            training_data_key: String identifying the private data to use for this model. This key must contain
            LabeledData (see: [Data](../../Data))

            test_split: percentage of test split
        """
        labeled_data = self._private_data.get(self._federated_data_identifier)
        length = len(labeled_data.data)
        train_data = labeled_data.data[int(test_split * length):]
        train_label = labeled_data.label[int(test_split * length):]
        test_data = labeled_data.data[:int(test_split * length)]
        test_label = labeled_data.label[:int(test_split * length)]

        self.set_private_data(LabeledData(train_data, train_label))
        self.set_private_test_data(LabeledData(test_data, test_label))


class FederatedData:
    """
    Class representing data across different data nodes. It supports Differential Privacy

    This object is iterable over different data nodes.
    """

    def __init__(self, epsilon_delta=None):
        self._data_nodes = []
        self._epsilon_delta = epsilon_delta

    def __getitem__(self, item):
        return self._data_nodes[item]

    def __iter__(self):
        return iter(self._data_nodes)

    def add_data_node(self, data):
        """
        This method adds a new node containing data to the federated data

        # Arguments:
            data: Data to add to this node
        """
        node = FederatedDataNode(str(id(self)), self._epsilon_delta)
        node.set_private_data(data)
        self._data_nodes.append(node)

    def num_nodes(self):
        """
        # Returns:
            num_nodes: The number of nodes in this federated data.
        """
        return len(self._data_nodes)

    def configure_data_access(self, data_access_definition):
        """
        Creates the same policy to access data over all the data nodes

        # Arguments:
            data_access_definition: (see: [DataAccessDefinition](../data/#dataaccessdefinition))
        """
        for data_node in self._data_nodes:
            data_node.configure_data_access(data_access_definition)

    def query(self):
        """
        Queries over every node and returns the answer of every node in a list

        # Returns
            answer: List containing responses for every node
        """
        answer = []
        for data_node in self._data_nodes:
            answer.append(data_node.query())

        return answer


class FederatedTransformation(abc.ABC):
    """
    Interface defining the method for applying an operation over [FederatedData](./#federateddata-class)
    """
    @abc.abstractmethod
    def apply(self, data):
        """
        This method receives data to be modified and performs the required modifications over it.

        # Arguments:
            data: The object that has to be modified
        """


def federate_array(array, num_data_nodes, epsilon_delta=None):
    """
    Creates [FederatedData](./#federateddata-class) from an indexable array.

    The array will be divided using the first dimension. It supports Differential Privacy

    # Arguments:
        array: Indexable array with any number of dimensions
        num_data_nodes: Number of nodes to use
        epsilon_delta: Epsilon-delta privacy budget to be set for each node

    # Returns
        federated_array: [FederatedData](./#federateddata-class) with an array of size len(array)/num_data_nodes \
        in every node
    """
    split_size = len(array) / float(num_data_nodes)
    last = 0.0

    federated_array = FederatedData(epsilon_delta)
    while last < len(array):
        federated_array.add_data_node(array[int(last):int(last + split_size)])
        last = last + split_size

    return federated_array


def apply_federated_transformation(federated_data, federated_transformation):
    """
    Applies the federated transformation over this federated data.

    Original federated data will be modified.

    # Arguments:
        federated_data: [FederatedData](./#federateddata-class) to use in the transformation
        federated_transformation: [FederatedTransformation](./#federatedtransformation-class) that will be applied \
        over this data
    """
    for data_node in federated_data:
        data_node.apply_data_transformation(federated_transformation)


def split_train_test(federated_data, test_split=0.2):
    for data_node in federated_data:
        data_node.split_train_test(test_split)

