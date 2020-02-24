import numpy as np
import pytest

from shfl.data_distribution.data_distribution_iid import IidDataDistribution
from shfl.data_base.data_base import DataBase
from shfl.data_distribution.data_distribution import DataDistribution


class TestDataDistribution(DataDistribution):
    def __init__(self,database):
        super(TestDataDistribution, self).__init__(database)

    def make_data_federated(self, data, labels, num_nodes, percent, weights):
        pass


class TestDataBase(DataBase):
    def __init__(self):
        super(TestDataBase, self).__init__()

    def load_data(self):
        self._train_data = np.random.rand(200).reshape([40,5])
        self._test_data = np.random.rand(200).reshape([40,5])
        self._validation_data = np.random.rand(200).reshape([40,5])
        self._train_labels = np.random.randint(0,10,40)
        self._test_labels = np.random.randint(0,10,40)
        self._validation_labels = np.random.randint(0,10,40)


def test_data_distribution_private_data():
    data = TestDataBase()
    data.load_data()

    dt = IidDataDistribution(data)

    train_data_e, train_labels_e, validation_data_e, validation_labels_e, test_data_e, test_labels_e = data.data

    train_data_d, train_labels_d, validation_data_d, validation_labels_d, test_data_d, test_labels_d = dt._database.data

    assert np.array_equal(train_data_e.ravel(), train_data_d.ravel())
    assert np.array_equal(validation_data_e.ravel(), validation_data_d.ravel())
    assert np.array_equal(test_data_e.ravel(), test_data_d.ravel())
    assert np.array_equal(train_labels_e,train_labels_d)
    assert np.array_equal(validation_labels_e, validation_labels_d)
    assert np.array_equal(test_labels_e, test_labels_d)

from shfl.core.query import Get

def test_get_federated_data():
    data = TestDataBase()
    data.load_data()

    dt = IidDataDistribution(data)

    # Identifier and num nodes is checked in core test.
    # Percent and weight is checked in idd and no_idd test.
    # So here, we only test mistaken param.
    mistaken = 50
    num_nodes = 4
    federated_data, test_data, test_label = dt.get_federated_data("id001", num_nodes, mistaken=mistaken)

    X_c = []
    y_c = []
    for i in range(federated_data.num_nodes()):
        X_c.append(federated_data[i].query_private_data(Get(), "id001").data)
        y_c.append(federated_data[i].query_private_data(Get(), "id001").label)

    X_c = np.array(X_c)
    y_c = np.array(y_c)

    train_data, train_labels = dt._database.train
    validation_data, validation_labels = dt._database.validation

    X = np.concatenate([train_data, validation_data], axis=0)
    y = np.concatenate([train_labels, validation_labels], axis=0)

    num_mistaken = 0
    for i,node in enumerate(X_c):
        labels_node = []
        for data in node:
            assert data in X
            labels_node.append(y[np.where((data==X).all(axis=1))[0][0]])
        if not (labels_node == y_c[i]).all():
            num_mistaken = num_mistaken + 1

    assert num_mistaken / num_nodes * 100 == mistaken
    assert num_mistaken / num_nodes * 100 > 0


