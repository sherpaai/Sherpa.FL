import abc
from shfl.private.federated_operation import FederatedTransformation
import random
import numpy as np


class FederatedDataAttack(abc.ABC):
    """
    Interface defining method to apply an FederatedAttack over [FederatedData](../Federated Operation/#federateddata-class)
    """

    @abc.abstractmethod
    def apply_attack(self, data):
        """
        This method receives federated data to be modified and performs the required modifications \
        (federated_attack) over it simulating the adversarial attack.

        # Arguments:
            federated_data: The data of nodes that we attack
        """


class ShuffleNode(FederatedTransformation):
    """
    Implementation of Federated Transformation for shuffling labels of labeled data in order to implement \
    data poisoning attack.

    This class implements interface [FederatedTransformation](../Federated Operation/#federatedtransformation-class).

    """
    def apply(self, labeled_data):
        """
        Method that implements apply abstract method of [FederatedTransformation](../Federated Operation/#federatedtransformation-class) \
        shuffling labels of labeled_data
        """
        random.shuffle(labeled_data.label)


class FederatedPoisoningDataAttack(FederatedDataAttack):
    """
    Class representing poisoning data attack simulation. This simulation consists on shuffling \
    the labels of some nodes. For that purpose, it uses class [ShuffleNode](./#shufflenode-class).

    This class implements interface [FederatedDataAttack](./#federateddataattack-class).

    # Attributes:
        * **_percentage, _adversaries**

    # Arguments:
        percentage: percentage of nodes that are adversarial ones

    # Properties:
        adversaries: Returns adversaries value
    """

    def __init__(self, percentage):
        super().__init__()
        self._percentage = percentage
        self._adversaries = []

    @property
    def adversaries(self):
        return self._adversaries

    def apply_attack(self, federated_data):
        """
        Method that implements federated attack of data poisoning shuffling training labels of some nodes.

        # Arguments:
            federated_data: Instance of Federated Data [Federated Data](../federated_operation#federateddata-class)
        """
        num_nodes = federated_data.num_nodes()
        list_nodes = np.arange(num_nodes)
        self._adversaries = random.sample(list(list_nodes), k=int(self._percentage / 100 * num_nodes))
        boolean_adversaries = [1 if x in self._adversaries else 0 for x in list_nodes]

        for node, boolean in zip(federated_data, boolean_adversaries):
            if boolean:
                node.apply_data_transformation(ShuffleNode())
