from tensorflow.keras.datasets import fashion_mnist
from shfl.data_base.data_base import DataBase

import tensorflow as tf


class FashionMnist(DataBase):
    """
    Implementation for load FASHION-MNIST data

    # References
        [FASHION-MNIST dataset](https://keras.io/datasets/#fashion-mnist-database-of-fashion-articles)
    """
    def __init__(self):
        super(FashionMnist, self).__init__()

    def load_data(self):
        """
        Load data from fashion emnist package

        # Returns:
            all_data : train data, train label, test data and test labels
        """
        ((self._train_data, self._train_labels), (self._test_data, self._test_labels)) = fashion_mnist.load_data()

        self._train_labels = tf.keras.utils.to_categorical(self._train_labels)
        self._test_labels = tf.keras.utils.to_categorical(self._test_labels)

        self.shuffle()
        
        return self._train_data, self._train_labels, self._test_data, self._test_labels
