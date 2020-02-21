import numpy as np
import pytest

import shfl.data_base.data_base
from shfl.data_base.data_base import DataBase


class TestDataBase(DataBase):
    def __init__(self):
        super(TestDataBase, self).__init__()

    def load_data(self):
        self._train_data = np.random.rand(50).reshape([10,5])
        self._test_data = np.random.rand(50).reshape([10,5])
        self._validation_data = np.random.rand(50).reshape([10,5])
        self._train_labels = np.random.rand(10)
        self._test_labels = np.random.rand(10)
        self._validation_labels = np.random.rand(10)


def test_extract_validation_samples():
    data = np.random.rand(50).reshape([10,-1])
    labels = np.random.rand(10)
    dim = 4

    rest_data, rest_labels, validation_data, validation_labels = shfl.data_base.data_base.extract_validation_samples(data,labels,dim)

    ndata = np.concatenate([rest_data,validation_data])
    nlabels = np.concatenate([rest_labels,validation_labels])

    data_ravel = np.sort(data.ravel())
    ndata_ravel = np.sort(ndata.ravel())

    assert np.array_equal(data_ravel,ndata_ravel)
    assert np.array_equal(np.sort(labels),np.sort(nlabels))
    assert rest_data.shape[0] == data.shape[0]-dim
    assert rest_labels.shape[0] == labels.shape[0]-dim
    assert validation_data.shape[0] == dim
    assert validation_labels.shape[0] == dim

def test_data_base_shuffle_elements():
    data = TestDataBase()
    data.load_data()

    train_data_b, train_labels_b, validation_data_b, validation_labels_b, test_data_b, test_labels_b = data.data

    data.shuffle()

    train_data_a, train_labels_a, validation_data_a, validation_labels_a, test_data_a, test_labels_a = data.data

    train_data_b = np.sort(train_data_b.ravel())
    train_data_a = np.sort(train_data_a.ravel())
    assert np.array_equal(train_data_b,train_data_a)

    validation_data_b = np.sort(validation_data_b.ravel())
    validation_data_a = np.sort(validation_data_a.ravel())
    assert np.array_equal(validation_data_b, validation_data_a)

    test_data_b = np.sort(test_data_b.ravel())
    test_data_a = np.sort(test_data_a.ravel())
    assert np.array_equal(test_data_b, test_data_a)

    assert np.array_equal(np.sort(train_labels_b), np.sort(train_labels_a))
    assert np.array_equal(np.sort(validation_labels_b), np.sort(validation_labels_a))
    assert np.array_equal(np.sort(test_labels_b), np.sort(test_labels_a))

def test_data_base_shuffle_correct():
    data = TestDataBase()
    data.load_data()

    train_data_b, train_labels_b, validation_data_b, validation_labels_b, test_data_b, test_labels_b = data.data

    data.shuffle()

    train_data_a, train_labels_a, validation_data_a, validation_labels_a, test_data_a, test_labels_a = data.data

    assert (train_data_b == train_data_a).all() == False
    assert (validation_data_b == validation_data_a).all() == False
    assert (test_data_b == test_data_a).all() == False

def test_shuffle_wrong_call():
    data = TestDataBase()

    with pytest.raises(TypeError):
        data.shuffle()