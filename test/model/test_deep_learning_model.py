import numpy as np
from unittest.mock import Mock
import pytest

from shfl.model.deep_learning_model import DeepLearningModel, KerasDeepLearningModel


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

    sizes = [(30, 64, 64), (64, 10)]

    l1 = layer()
    l1.input_shape = sizes[0]
    l2 = layer()
    l2.output_shape = sizes[1]
    model.layers = [l1, l2]

    batch = 32
    epoch = 2
    dpl = TestDeepLearningModel(model, batch, epoch)

    assert dpl._model.id == model.id
    assert dpl._batch_size == batch
    assert dpl._epochs == epoch
    assert np.array_equal(dpl._data_shape, sizes[0][1:])
    assert np.array_equal(dpl._label_shape, sizes[1][1:])


def test_train_wrong_data():
    model = Mock()
    layer = Mock

    sizes = [(30, 24, 24), (24, 10)]

    l1 = layer()
    l1.input_shape = sizes[0]
    l2 = layer()
    l2.output_shape = sizes[1]
    model.layers = [l1, l2]

    batch = 32
    epoch = 2
    kdpl = KerasDeepLearningModel(model, batch, epoch)

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

    sizes = [(1, 24, 24), (24, 10)]

    l1 = layer()
    l1.input_shape = sizes[0]
    l2 = layer()
    l2.output_shape = sizes[1]
    model.layers = [l1, l2]

    batch = 32
    epoch = 2
    kdpm = KerasDeepLearningModel(model, batch, epoch)

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


def test_predict():
    model = Mock()
    layer = Mock

    sizes = [(1, 24, 24), (24, 10)]

    l1 = layer()
    l1.input_shape = sizes[0]
    l2 = layer()
    l2.output_shape = sizes[1]
    model.layers = [l1, l2]

    batch = 32
    epoch = 2
    kdpm = KerasDeepLearningModel(model, batch, epoch)

    num_data = 30
    data = np.array([np.random.rand(24, 24) for i in range(num_data)])

    kdpm.predict(data)

    kdpm._model.predict.assert_called_once_with(data, batch_size=batch)


def test_wrong_predict():
    model = Mock()
    layer = Mock

    sizes = [(1, 24, 24), (24, 10)]

    l1 = layer()
    l1.input_shape = sizes[0]
    l2 = layer()
    l2.output_shape = sizes[1]
    model.layers = [l1, l2]

    batch = 32
    epoch = 2
    kdpm = KerasDeepLearningModel(model, batch, epoch)

    num_data = 30
    data = np.array([np.random.rand(16, 16) for i in range(num_data)])

    with pytest.raises(TypeError):
        kdpm.predict(data)


def test_get_model_params():
    model = Mock()
    layer = Mock

    num_data = 30
    sizes = [(1, 24, 24), (24, num_data)]

    l1 = layer()
    l1.input_shape = sizes[0]
    l2 = layer()
    l2.output_shape = sizes[1]
    model.layers = [l1, l2]

    batch = 32
    epoch = 2
    kdpm = KerasDeepLearningModel(model, batch, epoch)

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
    l1.input_shape = sizes[0]
    l2 = layer()
    l2.output_shape = sizes[1]
    model.layers = [l1, l2]

    batch = 32
    epoch = 2
    kdpm = KerasDeepLearningModel(model, batch, epoch)

    params = np.random.rand(30)
    kdpm.set_model_params(params)

    kdpm._model.set_weights.assert_called_once_with(params)

