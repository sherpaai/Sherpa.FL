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
        pass

    def set_model_params(self, params):
        pass


def test_deep_learning_model_private_data():
    criterion = Mock()
    optimizer = Mock()

    input = Mock()
    input.in_channels = 5
    out = Mock()
    out.out_features = 10
    model = [input, out, -1]

    batch = 32
    epoch = 2
    metrics = [0, 1, 2, 3]
    device = 'device0'
    dpl = TestDeepLearningModel(model, criterion, optimizer, batch, epoch, metrics, device)

    assert np.array_equal(dpl._model, model)
    assert dpl._data_shape == 5
    assert dpl._labels_shape == 10
    assert dpl._criterion.id == criterion.id
    assert dpl._optimizer.id == optimizer.id
    assert dpl._batch_size == batch
    assert dpl._epochs == epoch
    assert np.array_equal(dpl._metrics, metrics)
    assert dpl._device == device


@patch('shfl.model.deep_learning_model_pt.torch')
@patch('shfl.model.deep_learning_model_pt.TensorDataset')
@patch('shfl.model.deep_learning_model_pt.DataLoader')
def test_pytorch_model_train(mock_dl, mock_tdt, mock_torch):
    criterion = Mock()
    optimizer = Mock()

    input = Mock()
    input.in_channels = 1
    out = Mock()
    out.out_features = 10
    model = [input, out, -1]

    batch = 32
    epoch = 2
    metrics = None
    device = 'cpu'
    kdpm = DeepLearningModelPyTorch(model, criterion, optimizer, batch, epoch, metrics, device)
    model = Mock()
    model_return = [1, 2, 3, 4, 5]
    model.return_value = model_return
    kdpm._model = model

    num_data = 5
    data = np.array([np.random.rand(24, 24) for i in range(num_data)])
    data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2]))
    labels = np.array([np.zeros(10) for i in range(num_data)])
    for l in labels:
        l[np.random.randint(0, len(l))] = 1

    element = []
    for el, la in zip(data, labels):
        x = Mock()
        x.to.return_value = el
        y = Mock()
        y.to.return_value = la

        element.append([x, y])
    mock_dl.return_value = element
    kdpm.train(data, labels)

    optimizer_calls = []
    model_calls = []
    criterion_calls = []
    for i in range(epoch):
        for elem in element:
            inputs, y_true = elem[0].to(), elem[1].to()
            optimizer_calls.extend([call.zero_grad(), call.step()])
            model_calls.extend([call(inputs), call.zero_grad()])
            criterion_calls.extend([call(model_return, mock_torch.argmax(y_true, -1)), call().backward()])

    kdpm._optimizer.assert_has_calls(optimizer_calls)
    kdpm._model.assert_has_calls(model_calls)
    kdpm._criterion.assert_has_calls(criterion_calls)


@patch('shfl.model.deep_learning_model_pt.torch')
@patch('shfl.model.deep_learning_model_pt.TensorDataset')
@patch('shfl.model.deep_learning_model_pt.DataLoader')
def test_predict(mock_dl, mock_tdt, mock_torch):
    criterion = Mock()
    optimizer = Mock()

    input = Mock()
    input.in_channels = 1
    out = Mock()
    out.out_features = 10
    model = [input, out, -1]

    batch = 32
    epoch = 1
    metrics = None
    device = 'cpu'
    kdpm = DeepLearningModelPyTorch(model, criterion, optimizer, batch, epoch, metrics, device)
    model = Mock()
    kdpm._model = model

    y_pred = Mock()
    y_pred.cpu().numpy.return_value = [1, 2, 3]
    mock_torch.argmax.return_value = y_pred

    num_data = 5
    data = np.array([np.random.rand(24, 24) for i in range(num_data)])
    data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2]))

    element = []
    for el in data:
        x = Mock()
        x.to.return_value = el

        element.append([x, -1])
    mock_dl.return_value = element
    y_pred_return = kdpm.predict(data)

    model_calls = []
    torch_calls = []
    res = []
    for elem in element:
        inputs = elem[0].to()
        model_calls.append(call(inputs))
        torch_calls.extend([call.argmax(model(inputs), -1), call.argmax().cpu(), call.argmax().cpu().numpy()])
        res.extend(y_pred.cpu().numpy.return_value)

    kdpm._model.assert_has_calls(model_calls)
    mock_torch.assert_has_calls(torch_calls)
    assert np.array_equal(res, y_pred_return)


@patch('shfl.model.deep_learning_model_pt.torch')
@patch('shfl.model.deep_learning_model_pt.TensorDataset')
@patch('shfl.model.deep_learning_model_pt.DataLoader')
def test_evaluate(mock_dl, mock_tdt, mock_torch):
    criterion = Mock()
    optimizer = Mock()

    input = Mock()
    input.in_channels = 1
    out = Mock()
    out.out_features = 10
    model = [input, out, -1]

    batch = 32
    epoch = 2
    metrics = {'aux': lambda x, y: -1}
    device = 'cpu'
    kdpm = DeepLearningModelPyTorch(model, criterion, optimizer, batch, epoch, metrics, device)
    model = Mock()
    y_pred_model = Mock()
    y_pred_model.cpu().numpy.return_value = [1, 2, 3]
    model.return_value = y_pred_model
    kdpm._model = model
    mock_torch.argmax().numpy.return_value = np.array([1, 2, 3, 4, 5])
    kdpm._criterion.return_value = np.float64(0.0)

    num_data = 5
    data = np.array([np.random.rand(24, 24) for i in range(num_data)])
    data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2]))
    labels = np.array([np.zeros(10) for i in range(num_data)])
    for l in labels:
        l[np.random.randint(0, len(l))] = 1

    element = []
    for el, la in zip(data, labels):
        x = Mock()
        x.to.return_value = el
        y = Mock()
        y.to.return_value = la

        element.append([x, y])
    mock_dl.return_value = element
    res_metrics = kdpm.evaluate(data, labels)

    model_calls = []
    y_pred = []
    for elem in element:
        inputs, y_true = elem[0].to(), elem[1].to()

        model_calls.extend([call(inputs), call().cpu(), call().cpu().numpy()])
        y_pred.extend(y_pred_model.cpu().numpy.return_value)

    kdpm._model.assert_has_calls(model_calls)
    kdpm._criterion.assert_called_once_with(mock_torch.from_numpy(np.array(y_pred)),
                                       mock_torch.argmax(mock_torch.from_numpy(labels), -1))
    assert np.array_equal([0, 1, -1], res_metrics)


@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.evaluate')
def test_performance(mock_evaluate):
    criterion = Mock()
    optimizer = Mock()

    input = Mock()
    input.in_channels = 1
    out = Mock()
    out.out_features = 10
    model = [input, out, -1]

    batch = 32
    epoch = 1
    metrics = None
    device = 'cpu'
    kdpm = DeepLearningModelPyTorch(model, criterion, optimizer, batch, epoch, metrics, device)
    model = Mock()
    kdpm._model = model

    mock_evaluate.return_value = [0, 1, 2, 3, 4]

    num_data = 5
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

    input = Mock()
    input.in_channels = 1
    out = Mock()
    out.out_features = 10
    model = [input, out, -1]

    batch = 32
    epoch = 1
    metrics = None
    device = 'cpu'
    kdpm = DeepLearningModelPyTorch(model, criterion, optimizer, batch, epoch, metrics, device)
    model = Mock()
    kdpm._model = model

    params = np.random.rand(30)
    weights = []
    for elem in params:
        m = Mock()
        m.cpu().data.numpy.return_value = elem
        weights.append(m)

    kdpm._model.parameters.return_value = weights

    parm = kdpm.get_model_params()

    kdpm._model.parameters.assert_called_once()
    assert np.array_equal(params, parm)


def side_effect(value):
    return value


@patch('shfl.model.deep_learning_model_pt.torch')
def test_set_weights(mock_torch):
    criterion = Mock()
    optimizer = Mock()

    input = Mock()
    input.in_channels = 1
    out = Mock()
    out.out_features = 10
    model = [input, out, -1]

    batch = 32
    epoch = 1
    metrics = None
    device = 'cpu'
    kdpm = DeepLearningModelPyTorch(model, criterion, optimizer, batch, epoch, metrics, device)
    model = Mock()
    kdpm._model = model
    model_params = [9, 5, 4, 8, 5, 6]
    m_model_params = []
    for elem in model_params:
        aux = Mock()
        aux.data = elem
        m_model_params.append(aux)
    kdpm._model.parameters.return_value = m_model_params

    mock_torch.from_numpy.side_effect = side_effect

    set_params = [0, 1, 2, 3, 4, 5]
    kdpm.set_model_params(set_params)

    torch_calls = []
    for elem in set_params:
        torch_calls.append(call.from_numpy(elem))

    new_model_params = [x.data for x in kdpm._model.parameters()]

    mock_torch.assert_has_calls(torch_calls)
    assert np.array_equal(new_model_params, set_params)


def test_wrong_data():
    criterion = Mock()
    optimizer = Mock()

    input = Mock()
    input.in_channels = 1
    out = Mock()
    out.out_features = 10
    model = [input, out, -1]

    batch = 32
    epoch = 1
    metrics = None
    device = 'cpu'
    kdpm = DeepLearningModelPyTorch(model, criterion, optimizer, batch, epoch, metrics, device)

    num_data = 5
    data = np.array([np.random.rand(24, 24) for i in range(num_data)])

    with pytest.raises(AssertionError):
        kdpm._check_data(data)


def test_wrong_labels():
    criterion = Mock()
    optimizer = Mock()

    input = Mock()
    input.in_channels = 1
    out = Mock()
    out.out_features = 10
    model = [input, out, -1]

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
