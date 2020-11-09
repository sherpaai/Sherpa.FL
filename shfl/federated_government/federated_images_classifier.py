from shfl.federated_government.federated_government import FederatedGovernment
from shfl.federated_aggregator.fedavg_aggregator import FedAvgAggregator
from shfl.data_distribution.data_distribution_iid import IidDataDistribution
from shfl.data_distribution.data_distribution_non_iid import NonIidDataDistribution
from shfl.private.federated_operation import apply_federated_transformation
from shfl.private.federated_operation import FederatedTransformation
from shfl.model.deep_learning_model import DeepLearningModel
from shfl.data_base.emnist import Emnist
from shfl.data_base.fashion_mnist import FashionMnist
from shfl.private.federated_operation import Normalize

from enum import Enum
import numpy as np
import tensorflow as tf


class Reshape(FederatedTransformation):
    """
    Federated transformation to reshape the data
    """
    def apply(self, labeled_data):
        labeled_data.data = np.reshape(labeled_data.data,
                                       (labeled_data.data.shape[0], labeled_data.data.shape[1],
                                        labeled_data.data.shape[2], 1))


class ImagesDataBases(Enum):
    """
    Enumeration of possible databases for image classification.
    """
    EMNIST = Emnist
    FASHION_EMNIST = FashionMnist


class FederatedImagesClassifier(FederatedGovernment):
    """
    Class used to represent a high-level federated image classification
    (see: [FederatedGoverment](../federated_government/#federatedgovernment-class)).

    # Arguments:
        data_base_name_key: key of the enumeration of valid data bases (see: [ImagesDataBases](./#imagesdatabases-class))
        iid: boolean which specifies if the distribution if IID (True) or non-IID (False) (True by default)
        num_nodes: number of clients.
        percent: percentage of the database to distribute among nodes.
    """

    def __init__(self, data_base_name_key, iid=True, num_nodes=20, percent=100):
        if data_base_name_key in ImagesDataBases.__members__.keys():
            module = ImagesDataBases.__members__[data_base_name_key].value
            data_base = module()
            train_data, train_labels, test_data, test_labels = data_base.load_data()

            if iid:
                distribution = IidDataDistribution(data_base)
            else:
                distribution = NonIidDataDistribution(data_base)

            federated_data, self._test_data, self._test_labels = distribution.get_federated_data(num_nodes=num_nodes,
                                                                                                 percent=percent)
            apply_federated_transformation(federated_data, Reshape())
            mean = np.mean(train_data.data)
            std = np.std(train_data.data)
            apply_federated_transformation(federated_data, Normalize(mean, std))

            aggregator = FedAvgAggregator()

            super().__init__(self.model_builder, federated_data, aggregator)

        else:
            print("The data base name is not included. Try with: " + str(", ".join([e.name for e in ImagesDataBases])))
            self._test_data = None

    def run_rounds(self, n=5):
        """
        Overriding of the method of run_rounds of [FederatedGoverment](../federated_government/#federatedgovernment-class)).

        Run one more round beginning in the actual state testing in test data and federated_local_test.

        # Arguments:
            n: Number of rounds (2 by default)
        """
        if self._test_data is not None:
            self._test_data = np.reshape(self._test_data, (self._test_data.shape[0],
                                                           self._test_data.shape[1], self._test_data.shape[2], 1))
            for i in range(0, n):
                print("Accuracy round " + str(i))
                self.deploy_central_model()
                self.train_all_clients()
                self.evaluate_clients(self._test_data, self._test_labels)
                self.aggregate_weights()
                self.evaluate_global_model(self._test_data, self._test_labels)
                print("\n\n")
        else:
            print("Federated images classifier is not properly initialised")

    @staticmethod
    def model_builder():
        """
        Create a Tensorflow Model for image classification.

        # Returns:
            model: Instance of DeepLearningModel [DeepLearningModel](../model/#deeplearningmodel-class)).
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', strides=1,
                                         input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', strides=1))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        criterion = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.RMSprop()
        metrics = [tf.keras.metrics.CategoricalAccuracy()]

        return DeepLearningModel(model=model, criterion=criterion, optimizer=optimizer, metrics=metrics)
