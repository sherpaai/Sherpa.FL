import numpy as np
import random
import tensorflow as tf

from shfl.data_base.data_base import DataBase
from shfl.data_distribution.data_distribution_non_iid import NonIidDataDistribution
from shfl.private.data import UnprotectedAccess


class TestDataBase(DataBase):
    def __init__(self):
        super(TestDataBase, self).__init__()

    def load_data(self):
        self._train_data = np.random.rand(500).reshape([100, 5])
        self._test_data = np.random.rand(250).reshape([50, 5])
        self._train_labels = tf.keras.utils.to_categorical(np.random.randint(0, 3, 100))
        self._test_labels = tf.keras.utils.to_categorical(np.random.randint(0, 3, 50))


def test_choose_labels():
    num_nodes = 3
    total_labels = 10

    data = TestDataBase()
    data.load_data()
    data_distribution = NonIidDataDistribution(data)

    random_labels = data_distribution.choose_labels(num_nodes, total_labels)
    all_labels = np.concatenate(random_labels)

    for node in random_labels:
        assert len(node) <= total_labels

    assert len(random_labels) == num_nodes
    assert ((all_labels >= 0) & (all_labels < total_labels)).all()


def test_make_data_federated():
    random.seed(123)
    np.random.seed(123)

    data = TestDataBase()
    data.load_data()
    data_distribution = NonIidDataDistribution(data)

    train_data, train_label = data_distribution._database.train

    num_nodes = 3
    percent = 60
    # weights = np.full(num_nodes, 1/num_nodes)
    weights = [0.5, 0.25, 0.25]
    federated_data, federated_label = data_distribution.make_data_federated(train_data,
                                                                            train_label,
                                                                            percent,
                                                                            num_nodes,
                                                                            weights)

    all_data = np.concatenate(federated_data)
    all_label = np.concatenate(federated_label)

    idx = []
    for data in all_data:
        idx.append(np.where((data == train_data).all(axis=1))[0][0])

    seed_weights = [30, 15, 15]
    for i, weight in enumerate(weights):
        assert federated_data[i].shape[0] == seed_weights[i]

    #assert all_data.shape[0] == 60
    assert num_nodes == len(federated_data) == len(federated_label)
    assert (np.sort(all_data.ravel()) == np.sort(train_data[idx, ].ravel())).all()
    assert (np.sort(all_label, 0) == np.sort(train_label[idx], 0)).all()


def test_make_data_federated_wrong_weights():
    random.seed(123)
    np.random.seed(123)

    data = TestDataBase()
    data.load_data()
    data_distribution = NonIidDataDistribution(data)

    train_data, train_label = data_distribution._database.train

    num_nodes = 3
    percent = 60
    # weights = np.full(num_nodes, 1/num_nodes)
    weights = [0.5, 0.55, 0.1]
    federated_data, federated_label = data_distribution.make_data_federated(train_data,
                                                                            train_label,
                                                                            percent,
                                                                            num_nodes,
                                                                            weights,
                                                                            'without_replacement')

    weights = np.array([float(i) / sum(weights) for i in weights])

    all_data = np.concatenate(federated_data)
    all_label = np.concatenate(federated_label)

    idx = []
    for data in all_data:
        idx.append(np.where((data == train_data).all(axis=1))[0][0])

    seed_weights = [26, 28, 5]
    for i, weight in enumerate(weights):
        assert federated_data[i].shape[0] == seed_weights[i]

    assert num_nodes == len(federated_data) == len(federated_label)
    assert (np.sort(all_data.ravel()) == np.sort(train_data[idx, ].ravel())).all()
    assert (np.sort(all_label, 0) == np.sort(train_label[idx], 0)).all()


def test_get_federated_data():
    data = TestDataBase()
    data.load_data()
    dt = NonIidDataDistribution(data)

    # Identifier and num nodes is checked in private test.
    # Percent and weight is checked in idd and no_idd test.
    num_nodes = 4
    federated_data, test_data, test_label = dt.get_federated_data(num_nodes)

    x_c = []
    y_c = []
    federated_data.configure_data_access(UnprotectedAccess())
    for i in range(federated_data.num_nodes()):
        x_c.append(federated_data[i].query().data)
        y_c.append(federated_data[i].query().label)

    x_c = np.array(x_c)
    x, y = dt._database.train

    y = tf.keras.utils.to_categorical(y)

    idx = []
    for i, node in enumerate(x_c):
        labels_node = []
        for data in node:
            assert data in x
            idx.append(np.where((data == x).all(axis=1))[0][0])
            labels_node.append(y[idx[-1]].argmax(axis=-1))

    assert np.array_equal(x[idx, ].ravel(), np.concatenate(x_c).ravel())
    assert np.array_equal(test_data.ravel(), dt._database.test[0].ravel())
    assert np.array_equal(test_label, dt._database.test[1])


