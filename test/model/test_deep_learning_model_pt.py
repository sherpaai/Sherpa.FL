import numpy as np
from unittest.mock import Mock, patch, call
import pytest

from shfl.model.deep_learning_model_pt import DeepLearningModelPyTorch


class TestDeepLearningModel(DeepLearningModelPyTorch):
    def train(self, data, labels):
        pass

    def predict(self, data):
        pass

    def get_model_params(self):
        return [np.random.rand(5, 1, 32, 32), np.random.rand(10, )]

    def set_model_params(self, params):
        pass


def test_deep_learning_model_private_data():
    criterion = Mock()
    optimizer = Mock()
    model = Mock()

    batch = 32
    epoch = 2
    metrics = [0, 1, 2, 3]
    device = 'device0'
    dpl = TestDeepLearningModel(model, criterion, optimizer, batch, epoch, metrics, device)

    assert dpl._model.id == model.id
    assert dpl._data_shape == 1
    assert dpl._labels_shape == (10,)
    assert dpl._criterion.id == criterion.id
    assert dpl._optimizer.id == optimizer.id
    assert dpl._batch_size == batch
    assert dpl._epochs == epoch
    assert np.array_equal(dpl._metrics, metrics)
    assert dpl._device == device


@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.get_model_params')
@patch('shfl.model.deep_learning_model_pt.torch')
@patch('shfl.model.deep_learning_model_pt.TensorDataset')
@patch('shfl.model.deep_learning_model_pt.DataLoader')
def test_pytorch_model_train(mock_dl, mock_tdt, mock_torch, mock_get_params):
    criterion = Mock()
    optimizer = Mock()

    model = Mock()
    model_return = [1, 2, 3, 4, 5]
    model.return_value = model_return

    mock_get_params.return_value = [np.random.rand(5, 1, 24, 24), np.random.rand(10)]

    batch = 1
    epoch = 2
    metrics = None
    device = 'cpu'
    kdpm = DeepLearningModelPyTorch(model, criterion, optimizer, batch, epoch, metrics, device)

    num_data = 5
    data = np.array([np.random.rand(24, 24) for i in range(num_data)])
    data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2]))
    labels = np.array([np.zeros(10) for i in range(num_data)])
    for l in labels:
        l[np.random.randint(0, len(l))] = 1

    element = []
    for el, la in zip(data, labels):
        x = Mock()
        x.float().to.return_value = el[np.newaxis]
        y = Mock()
        y.float().to.return_value = la[np.newaxis]

        element.append([x, y])
    mock_dl.return_value = element
    kdpm.train(data, labels)

    optimizer_calls = []
    model_calls = []
    criterion_calls = []
    for i in range(epoch):
        for elem in element:
            inputs, y_true = elem[0].float().to(), elem[1].float().to()
            optimizer_calls.extend([call.zero_grad(), call.step()])
            model_calls.extend([call(inputs), call.zero_grad()])
            criterion_calls.extend([call(model_return, mock_torch.argmax(y_true, -1)), call().backward()])

    kdpm._optimizer.assert_has_calls(optimizer_calls)
    kdpm._model.assert_has_calls(model_calls)
    kdpm._criterion.assert_has_calls(criterion_calls)


@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.get_model_params')
@patch('shfl.model.deep_learning_model_pt.torch')
@patch('shfl.model.deep_learning_model_pt.TensorDataset')
@patch('shfl.model.deep_learning_model_pt.DataLoader')
def test_predict(mock_dl, mock_tdt, mock_torch, mock_get_params):
    criterion = Mock()
    optimizer = Mock()

    model = Mock()
    model_return = Mock()
    model_return.cpu().numpy.return_value = [1, 2, 3, 4]
    model.return_value = model_return

    mock_get_params.return_value = [np.random.rand(5, 1, 24, 24), np.random.rand(10)]

    batch = 32
    epoch = 1
    metrics = None
    device = 'cpu'
    kdpm = DeepLearningModelPyTorch(model, criterion, optimizer, batch, epoch, metrics, device)

    num_data = 5
    data = np.array([np.random.rand(24, 24) for i in range(num_data)])
    data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2]))

    element = []
    for el in data:
        x = Mock()
        x.float().to.return_value = el[np.newaxis]

        element.append([x, -1])
    mock_dl.return_value = element
    y_pred_return = kdpm.predict(data)

    model_calls = []
    res = []
    for elem in element:
        inputs = elem[0].float().to()
        model_calls.extend([call(inputs), call(inputs).cpu(), call(inputs).cpu().numpy()])
        res.extend(model_return.cpu().numpy.return_value)

    kdpm._model.assert_has_calls(model_calls)
    assert np.array_equal(res, y_pred_return)


def side_effect_from_numpy(value):
    x = Mock()
    x.float.return_value = value

    return x


def side_effect_argmax(value, axis):
    return value


@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.predict')
@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.get_model_params')
@patch('shfl.model.deep_learning_model_pt.torch')
def test_evaluate(mock_torch, mock_get_params, mock_predict):
    num_data = 5
    criterion = Mock()
    optimizer = Mock()
    criterion.return_value = np.float64(0.0)

    model = Mock()

    mock_torch.argmax.side_effect = side_effect_argmax
    mock_torch.from_numpy.side_effect = side_effect_from_numpy

    predict_return = Mock()
    predict_return.cpu().numpy.return_value = np.random.rand(5, 10)
    mock_predict.return_value = predict_return

    mock_get_params.return_value = [np.random.rand(num_data, 1, 24, 24), np.random.rand(10)]

    batch = 32
    epoch = 2
    metrics = {'aux': lambda x, y: -1}
    device = 'cpu'
    kdpm = DeepLearningModelPyTorch(model, criterion, optimizer, batch, epoch, metrics, device)

    data = np.array([np.random.rand(24, 24) for i in range(num_data)])
    data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2]))
    labels = np.array([np.zeros(10) for i in range(num_data)])
    for l in labels:
        l[np.random.randint(0, len(l))] = 1

    res_metrics = kdpm.evaluate(data, labels)

    mock_predict.assert_called_once_with(data)
    kdpm._criterion.assert_called_once_with(mock_predict.return_value, labels)
    assert np.array_equal([0, -1], res_metrics)


@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.evaluate')
@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.get_model_params')
def test_performance(mock_get_params, mock_evaluate):
    num_data = 5
    criterion = Mock()
    optimizer = Mock()
    model = Mock()
    criterion.return_value = np.float64(0.0)

    mock_get_params.return_value = [np.random.rand(num_data, 1, 24, 24), np.random.rand(10)]

    mock_evaluate.return_value = [0, 1, 2, 3, 4]

    batch = 32
    epoch = 1
    metrics = None
    device = 'cpu'
    kdpm = DeepLearningModelPyTorch(model, criterion, optimizer, batch, epoch, metrics, device)

    data = np.array([np.random.rand(24, 24) for i in range(num_data)])
    data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2]))
    labels = np.array([np.zeros(10) for i in range(num_data)])
    for l in labels:
        l[np.random.randint(0, len(l))] = 1

    res = kdpm.performance(data, labels)

    mock_evaluate.assert_called_once_with(data, labels)
    assert res == mock_evaluate.return_value[0]


def test_get_model_params():
    criterion = Mock()
    optimizer = Mock()

    model = Mock()
    params = [np.random.rand(5, 1, 2) for i in range(5)]
    params.append(np.random.rand(10))
    weights = []
    for elem in params:
        m = Mock()
        m.cpu().data.numpy.return_value = elem
        weights.append(m)
    model.parameters.return_value = weights

    batch = 32
    epoch = 1
    metrics = None
    device = 'cpu'
    kdpm = DeepLearningModelPyTorch(model, criterion, optimizer, batch, epoch, metrics, device)

    parm = kdpm.get_model_params()

    # two calls in constructor and one call in get_model_params method
    kdpm._model.parameters.assert_has_calls([call() for i in range(3)])
    for one, two in zip(params, parm):
        assert np.array_equal(one, two)


@patch('shfl.model.deep_learning_model_pt.torch')
@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.get_model_params')
def test_set_weights(mock_get_params, mock_torch):
    num_data = 5
    criterion = Mock()
    optimizer = Mock()
    criterion.return_value = np.float64(0.0)

    model = Mock()
    model_params = [9, 5, 4, 8, 5, 6]
    m_model_params = []
    for elem in model_params:
        aux = Mock()
        aux.data = elem
        m_model_params.append(aux)
    model.parameters.return_value = m_model_params

    mock_get_params.return_value = [np.random.rand(num_data, 1, 24, 24), np.random.rand(10)]

    mock_torch.from_numpy.side_effect = side_effect_from_numpy

    batch = 32
    epoch = 1
    metrics = None
    device = 'cpu'
    kdpm = DeepLearningModelPyTorch(model, criterion, optimizer, batch, epoch, metrics, device)

    set_params = [0, 1, 2, 3, 4, 5]
    kdpm.set_model_params(set_params)

    new_model_params = [x.data for x in kdpm._model.parameters()]

    assert np.array_equal(new_model_params, set_params)


@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.get_model_params')
def test_wrong_data(mock_get_params):
    num_data = 5
    criterion = Mock()
    optimizer = Mock()
    model = Mock()
    criterion.return_value = np.float64(0.0)

    mock_get_params.return_value = [np.random.rand(num_data, 1, 24, 24), np.random.rand(10)]

    batch = 32
    epoch = 1
    metrics = None
    device = 'cpu'
    kdpm = DeepLearningModelPyTorch(model, criterion, optimizer, batch, epoch, metrics, device)

    num_data = 5
    data = np.array([np.random.rand(24, 24) for i in range(num_data)])

    with pytest.raises(AssertionError):
        kdpm._check_data(data)


@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.get_model_params')
def test_wrong_labels(mock_get_params):
    num_data = 5
    criterion = Mock()
    optimizer = Mock()
    model = Mock()
    criterion.return_value = np.float64(0.0)

    mock_get_params.return_value = [np.random.rand(num_data, 1, 24, 24), np.random.rand(10)]

    batch = 32
    epoch = 1
    metrics = None
    device = 'cpu'
    kdpm = DeepLearningModelPyTorch(model, criterion, optimizer, batch, epoch, metrics, device)

    num_data = 5
    labels = np.array([np.zeros(9) for i in range(num_data)])
    for l in labels:
        l[np.random.randint(0, len(l))] = 1

    with pytest.raises(AssertionError):
        kdpm._check_labels(labels)
