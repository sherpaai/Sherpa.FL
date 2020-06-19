import numpy as np
import emnist

from shfl.data_base import data_base as db


class Emnist(db.DataBase):
    """
    Implementation for load EMNIST data

    # References
        [EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
    """
    def __init__(self):
        super(Emnist, self).__init__()

    def load_data(self):
        """
        Load data from emnist package

        # Returns:
            all_data : train data, train labels, test data and test labels
        """
        self._train_data, self._train_labels = emnist.extract_training_samples('digits')
        self._train_labels = np.eye(10)[self._train_labels]
        self._test_data, self._test_labels = emnist.extract_test_samples('digits')
        self._test_labels = np.eye(10)[self._test_labels]

        self.shuffle()

        return self.data
