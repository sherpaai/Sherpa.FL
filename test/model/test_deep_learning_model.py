import numpy as np
from unittest.mock import Mock
import pytest
import tensorflow as tf

from shfl.model.deep_learning_model import DeepLearningModel


class TestDeepLearningModel(DeepLearningModel):
    def train(self, data, labels):
        pass

    def predict(self, data):
        pass

    def get_model_params(self):
        pass

    def set_model_params(self, params):
        pass


def test_deep_learning_model_private_data():
    model = Mock()
    layer = Mock
    criterion = Mock()
    optimizer = Mock()
    metrics = Mock()

    sizes = [(30, 64, 64), (64, 10)]

    l1 = layer()
    l1.get_input_shape_at.return_value = sizes[0]
    l2 = layer()
    l2.get_output_shape_at.return_value = sizes[1]
    model.layers = [l1, l2]

    batch = 32
    epoch = 2
    dpl = TestDeepLearningModel(model, criterion, optimizer, batch, epoch, metrics)

    dpl._model.compile.assert_called_once_with(optimizer=dpl._optimizer, loss=dpl._criterion, metrics=dpl._metrics)

    assert dpl._model.id == model.id
    assert dpl._batch_size == batch
    assert dpl._epochs == epoch
    assert np.array_equal(dpl._data_shape, sizes[0][1:])
    assert np.array_equal(dpl._labels_shape, sizes[1][1:])
    assert dpl._criterion.id == criterion.id
    assert dpl._optimizer.id == optimizer.id


def test_train_wrong_data():
    model = Mock()
    layer = Mock
    criterion = Mock()
    optimizer = Mock()
    metrics = Mock()

    sizes = [(30, 24, 24), (24, 10)]

    l1 = layer()
    l1.get_input_shape_at.return_value = sizes[0]
    l2 = layer()
    l2.get_output_shape_at.return_value = sizes[1]
    model.layers = [l1, l2]

    batch = 32
    epoch = 2
    kdpl = DeepLearningModel(model, criterion, optimizer, batch, epoch, metrics)

    num_data = 30
    data = np.array([np.random.rand(16, 16) for i in range(num_data)])
    label = np.array([np.zeros(10) for i in range(num_data)])
    for l in label:
        l[np.random.randint(0, len(l))] = 1

    with pytest.raises(AssertionError):
        kdpl.train(data, label)

    data = np.array([np.random.rand(24, 24) for i in range(num_data)])
    label = np.array([np.zeros(8) for i in range(num_data)])
    for l in label:
        l[np.random.randint(0, len(l))] = 1
    with pytest.raises(AssertionError):
        kdpl.train(data, label)


def test_keras_model_train():
    model = Mock()
    layer = Mock
    criterion = Mock()
    optimizer = Mock()
    metrics = Mock()

    sizes = [(1, 24, 24), (24, 10)]

    l1 = layer()
    l1.get_input_shape_at.return_value = sizes[0]
    l2 = layer()
    l2.get_output_shape_at.return_value = sizes[1]
    model.layers = [l1, l2]

    batch = 32
    epoch = 2
    kdpm = DeepLearningModel(model, criterion, optimizer, batch, epoch, metrics)

    num_data = 30
    data = np.array([np.random.rand(24, 24) for i in range(num_data)])
    labels = np.array([np.zeros(10) for i in range(num_data)])
    for l in labels:
        l[np.random.randint(0, len(l))] = 1

    kdpm.train(data, labels)

    kdpm._model.fit.assert_called_once()
    params = kdpm._model.fit.call_args_list[0][1]

    assert np.array_equal(params['x'], data)
    assert np.array_equal(params['y'], labels)
    assert params['batch_size'] == batch
    assert params['epochs'] == epoch


def test_evaluate():
    model = Mock()
    layer = Mock
    criterion = Mock()
    optimizer = Mock()
    metrics = Mock()

    sizes = [(1, 24, 24), (24, 10)]

    l1 = layer()
    l1.get_input_shape_at.return_value = sizes[0]
    l2 = layer()
    l2.get_output_shape_at.return_value = sizes[1]
    model.layers = [l1, l2]

    batch = 32
    epoch = 2
    kdpm = DeepLearningModel(model, criterion, optimizer, batch, epoch, metrics)

    num_data = 30
    data = np.array([np.random.rand(24, 24) for i in range(num_data)])
    labels = np.array([np.zeros(10) for i in range(num_data)])
    for l in labels:
        l[np.random.randint(0, len(l))] = 1

    kdpm.evaluate(data, labels)

    kdpm._model.evaluate.assert_called_once()


def test_predict():
    model = Mock()
    layer = Mock
    criterion = Mock()
    optimizer = Mock()
    metrics = Mock()

    sizes = [(1, 24, 24), (24, 10)]

    l1 = layer()
    l1.get_input_shape_at.return_value = sizes[0]
    l2 = layer()
    l2.get_output_shape_at.return_value = sizes[1]
    model.layers = [l1, l2]

    batch = 32
    epoch = 2
    kdpm = DeepLearningModel(model, criterion, optimizer, batch, epoch, metrics)

    num_data = 30
    data = np.array([np.random.rand(24, 24) for i in range(num_data)])

    kdpm.predict(data)

    kdpm._model.predict.assert_called_once_with(data, batch_size=batch)


def test_wrong_predict():
    model = Mock()
    layer = Mock
    criterion = Mock()
    optimizer = Mock()
    metrics = Mock()

    sizes = [(1, 24, 24), (24, 10)]

    l1 = layer()
    l1.get_input_shape_at.return_value = sizes[0]
    l2 = layer()
    l2.get_output_shape_at.return_value = sizes[1]
    model.layers = [l1, l2]

    batch = 32
    epoch = 2
    kdpm = DeepLearningModel(model, criterion, optimizer, batch, epoch, metrics)

    num_data = 30
    data = np.array([np.random.rand(16, 16) for i in range(num_data)])

    with pytest.raises(AssertionError):
        kdpm.predict(data)


def test_get_model_params():
    model = Mock()
    layer = Mock

    num_data = 30
    sizes = [(1, 24, 24), (24, num_data)]

    l1 = layer()
    l1.get_input_shape_at.return_value = sizes[0]
    l2 = layer()
    l2.get_output_shape_at.return_value = sizes[1]
    model.layers = [l1, l2]

    batch = 32
    epoch = 2
    kdpm = DeepLearningModel(model, batch, epoch)

    params = np.random.rand(30)
    kdpm._model.get_weights.return_value = params
    parm = kdpm.get_model_params()

    assert np.array_equal(params, parm)


def test_set_weights():
    model = Mock()
    layer = Mock

    num_data = 30
    sizes = [(1, 24, 24), (24, num_data)]

    l1 = layer()
    l1.get_input_shape_at.return_value = sizes[0]
    l2 = layer()
    l2.get_output_shape_at.return_value = sizes[1]
    model.layers = [l1, l2]

    batch = 32
    epoch = 2
    kdpm = DeepLearningModel(model, batch, epoch)

    params = np.random.rand(30)
    kdpm.set_model_params(params)

    kdpm._model.set_weights.assert_called_once_with(params)


def test_performance():
    model = Mock()
    layer = Mock

    num_data = 30
    sizes = [(1, 24, 24), (24, num_data)]

    l1 = layer()
    l1.get_input_shape_at.return_value = sizes[0]
    l2 = layer()
    l2.get_output_shape_at.return_value = sizes[1]
    model.layers = [l1, l2]

    model.evaluate.return_value = [0, 0]

    batch = 32
    epoch = 2
    kdpm = DeepLearningModel(model, batch, epoch)

    kdpm._check_data = Mock()
    kdpm._check_labels = Mock()

    data = np.random.rand(25).reshape((5, 5))
    labels = np.random.randint(0, 2, 5)

    res = kdpm.performance(data, labels)

    kdpm._check_data.assert_called_once_with(data)
    kdpm._check_labels.assert_called_once_with(labels)

    kdpm._model.evaluate.assert_called_once_with(data, labels, verbose=0)

    assert res == 0

