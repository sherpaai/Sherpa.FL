import abc
from shfl.core.node import DataNode


class FederatedData:
    """
    Class representing data across different data nodes.

    Every identifier for FederatedData objects only can be used once.

    Attributes
    ----------
    _data_nodes : list
        List containing data nodes that are part of the federated data
    _identifier : str
        Unique identifier for the federated data
    """

    __used_identifiers = set()

    def __init__(self, identifier):
        if identifier in FederatedData.__used_identifiers:
            raise ValueError("Identifier " + str(identifier) + "is already in use")
        self._data_nodes = []
        self._identifier = identifier
        FederatedData.__used_identifiers.add(identifier)

    def __del__(self):
        FederatedData.__used_identifiers.remove(self._identifier)

    def __getitem__(self, item):
        return self._data_nodes[item]

    def __iter__(self):
        return iter(self._data_nodes)

    @property
    def identifier(self):
        return self._identifier

    def add_data_node(self, node, data):
        node.set_private_data(self._identifier, data)
        self._data_nodes.append(node)

    def num_nodes(self):
        return len(self._data_nodes)


class FederatedTransformation(abc.ABC):
    """
    Interface defining method to apply an operation over FederatedData
    """
    @abc.abstractmethod
    def apply(self, data):
        """
        Parameters
        ----------
        data : object
            The object that have to be modified
        """


def apply_federated_transformation(federated_data, federated_transformation):
    """
    Applies the federated transformation over this federated data

    Original federated data will be modified.

    Parameters
    ----------
    federated_data : ~FederatedData
        ~FederatedData to use in the transformation
    federated_transformation : ~FederatedTransformation
        ~FederatedTransformation that will be applied over this data

    """
    for data_node in federated_data:
        data_node.apply_data_transformation(federated_data.identifier, federated_transformation)


def query_federated_data(federated_data, query):
    """
    Apply the federated query over every node and returns the answer of every node in a list

    Parameters
    ----------
    federated_data : ~FederatedData
        ~FederatedData to use in the query
    query : ~Query
        ~Query that will be applied over this data

    Returns
    -------
    list
       List containing responses for every node
    """
    answer = []
    for data_node in federated_data:
        answer.append(data_node.query_private_data(query, federated_data.identifier))

    return answer


def federate_array(identifier, array, num_data_nodes):
    """
    Create ~FederatedData from a numpy array.

    The array will be divided using the first dimension.

    Parameters
    ----------
    identifier : str
        Unique identifier that will be used for the FederatedData
    array : numpy array
        Numpy array with any number of dimensions
    num_data_nodes: int
        Number of nodes to use

    Returns
    -------
    ~FederatedData
        FederatedData with an array of size len(array)/num_data_nodes in every node.
    """
    split_size = len(array) / float(num_data_nodes)
    last = 0.0

    federated_array = FederatedData(identifier)
    while last < len(array):
        federated_array.add_data_node(DataNode(), array[int(last):int(last + split_size)])
        last = last + split_size

    return federated_array
