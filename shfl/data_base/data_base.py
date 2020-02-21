import abc
import numpy as np


def extract_validation_samples(data, labels, dim):
    """
    Method that randomly choose the validation data from data and labels.

    Parameters
    ----------
    data: numpy matrix
        data for extract the validation data
    labels: numpy array
        Labels from data
    dim: int
        Size for validation data

    Return
    ------
    new_data : list
        Set of new data, new labels, validation data and validation labels
    """
    randomize = np.arange(len(labels))
    np.random.shuffle(randomize)
    data = data[randomize, ]
    labels = labels[randomize]

    validation_data = data[0:dim, ]
    validation_labels = labels[0:dim]

    rest_data = data[dim:, ]
    rest_labels = labels[dim:]

    return rest_data, rest_labels, validation_data, validation_labels


class DataBase(abc.ABC):
    """
    Interface for data base

    Attributes
    ----------
    _train_data : numpy matrix
        Data train
    _train_labels : numpy array
        Label for each train element
    _test_data : numpy matrix
        Data test
    _test_labels : numpy array
        Label for each test element
    _validation_data : numpy matrix
        Data from validation
    _validation_labels : numpy array
        Label for each validation data element
    """

    def __init__(self):
        super(DataBase, self).__init__()
        self._train_data = []
        self._test_data = []
        self._validation_data = []
        self._train_labels = []
        self._test_labels = []
        self._validation_labels = []

    @property
    def train(self):
        """
        Returns train data and labels
        -------
        """
        return self._train_data, self._train_labels

    @property
    def validation(self):
        """
        Returns validation data and labels
        -------
        """
        return self._validation_data, self._validation_labels

    @property
    def test(self):
        """
        Returns validation data and labels
        -------
        """
        return self._test_data, self._test_labels

    @property
    def data(self):
        """
        Property method for data

        Return
        ------
        all_data : list
            Set of train data, label train, validation data, validation train, test data and label test
        """
        return self._train_data, self._train_labels, self._validation_data, self._validation_labels, \
            self._test_data, self._test_labels

    @abc.abstractmethod
    def load_data(self):
        """
        Abstract method that load the data
        """

    def shuffle(self):
        """
        Shuffle all data
        """
        randomize = np.arange(len(self._train_labels))
        np.random.shuffle(randomize)
        self._train_data = self._train_data[randomize, ]
        self._train_labels = self._train_labels[randomize]

        randomize = np.arange(len(self._test_labels))
        np.random.shuffle(randomize)
        self._test_data = self._test_data[randomize, ]
        self._test_labels = self._test_labels[randomize]

        randomize = np.arange(len(self._validation_labels))
        np.random.shuffle(randomize)
        self._validation_data = self._validation_data[randomize,]
        self._validation_labels = self._validation_labels[randomize]
