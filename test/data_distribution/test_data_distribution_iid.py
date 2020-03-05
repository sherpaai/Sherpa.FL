import numpy as np

from shfl.data_base.data_base import DataBase
from shfl.data_distribution.data_distribution_iid import IidDataDistribution


class TestDataBase(DataBase):
    def __init__(self):
        super(TestDataBase, self).__init__()

    def load_data(self):
        self._train_data = np.random.rand(200).reshape([40, 5])
        self._test_data = np.random.rand(200).reshape([40, 5])
        self._validation_data = np.random.rand(200).reshape([40, 5])
        self._train_labels = np.random.randint(0, 10, 40)
        self._test_labels = np.random.randint(0, 10, 40)
        self._validation_labels = np.random.randint(0, 10, 40)


def test_make_data_federated():
    data = TestDataBase()
    data.load_data()
    data_distribution = IidDataDistribution(data)

    train_data, train_label = data_distribution._database.train
    validation_data, validation_label = data_distribution._database.validation

    train_data = np.concatenate((train_data, validation_data), axis=0)
    train_label = np.concatenate((train_label, validation_label), axis=0)

    num_nodes = 3
    percent = 60
    # weights = np.full(num_nodes, 1/num_nodes)
    weights = [0.5, 0.25, 0.25]
    federated_data, federated_label = data_distribution.make_data_federated(train_data,
                                                                            train_label,
                                                                            num_nodes,
                                                                            percent,
                                                                            weights)
    data_distribution.get_federated_data("id000", 3)

    all_data = np.concatenate(federated_data)
    all_label = np.concatenate(federated_label)

    idx = []
    for data in all_data:
        idx.append(np.where((data == train_data).all(axis=1))[0][0])

    for i, weight in enumerate(weights):
        assert federated_data[i].shape[0] == int(weight * int(percent * train_data.shape[0] / 100))

    assert all_data.shape[0] == int(percent * train_data.shape[0] / 100)
    assert num_nodes == federated_data.shape[0] == federated_label.shape[0]
    assert (np.sort(all_data.ravel()) == np.sort(train_data[idx,].ravel())).all()
    assert (np.sort(all_label) == np.sort(train_label[idx])).all()

    #test make federated data with replacement
    federated_data, federated_label = data_distribution.make_data_federated(train_data,
                                                                            train_label,
                                                                            num_nodes,
                                                                            percent,
                                                                            weights,
                                                                            sampling="with_replacement")
    all_data = np.concatenate(federated_data)
    all_label = np.concatenate(federated_label)

    idx = []
    for data in all_data:
        idx.append(np.where((data == train_data).all(axis=1))[0][0])

    for i, weight in enumerate(weights):
        assert federated_data[i].shape[0] == int(weight * int(percent * train_data.shape[0] / 100))

    assert all_data.shape[0] == int(percent * train_data.shape[0] / 100)
    assert num_nodes == federated_data.shape[0] == federated_label.shape[0]
    assert (np.sort(all_data.ravel()) == np.sort(train_data[idx,].ravel())).all()
    assert (np.sort(all_label) == np.sort(train_label[idx])).all()
