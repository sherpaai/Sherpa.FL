from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from shfl.data_base.data_base import DataBase

import tensorflow as tf

class Cifar10(DataBase):
    """
    Implementation for load CIFAR10 data

    # References
        [CIFAR10 dataset](https://keras.io/api/datasets/cifar10)
    """
    def __init__(self):
        super(Cifar10, self).__init__()

    def load_data(self):
        """
        Load data from CIFAR10 package

        # Returns:
            all_data : train data, train label, test data and test labels
        """
        ((self._train_data, self._train_labels), (self._test_data, self._test_labels)) = cifar10.load_data()

        self._train_labels = tf.keras.utils.to_categorical(self._train_labels)
        self._test_labels = tf.keras.utils.to_categorical(self._test_labels)

        self.shuffle()
        
        return self._train_data, self._train_labels, self._test_data, self._test_labels

class Cifar100(DataBase):
    """
    Implementation for load CIFAR100 data

    # References
        [CIFAR100 dataset](https://keras.io/api/datasets/cifar100)
    """
    def __init__(self):
        super(Cifar100, self).__init__()

    def load_data(self):
        """
        Load data from CIFAR100 package

        # Returns:
            all_data : train data, train label, test data and test labels
        """
        ((self._train_data, self._train_labels), (self._test_data, self._test_labels)) = cifar100.load_data()

        self._train_labels = tf.keras.utils.to_categorical(self._train_labels)
        self._test_labels = tf.keras.utils.to_categorical(self._test_labels)

        self.shuffle()
        
        return self._train_data, self._train_labels, self._test_data, self._test_labels