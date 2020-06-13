import numpy as np
from unittest.mock import Mock, patch
import pytest


from shfl.model.linear_regression_model import LinearRegressionModel


def test_linear_regression_model_initialize_single_target():
    n_features = 9
    n_targets = 1
    lnr = LinearRegressionModel(n_features = n_features, n_targets = n_targets)
    
    assert np.shape(lnr._model.intercept_) == ()
    assert np.shape(lnr._model.coef_) == (n_features,)
    assert np.shape(lnr.get_model_params()) == (n_targets, n_features + 1)


def test_linear_regression_model_initialize_multiple_targets():
    n_features = 9
    n_targets = 2
    lnr = LinearRegressionModel(n_features = n_features, n_targets = n_targets)
    
    assert np.shape(lnr._model.intercept_) == (n_targets,)
    assert np.shape(lnr._model.coef_) == (n_targets, n_features)
    assert np.shape(lnr.get_model_params()) == (n_targets, n_features + 1)
    
def test_linear_regression_model_wrong_initialization():
    n_features = [9.5, -1, 9, 9] 
    n_targets = [1, 1, -1, 1.5]
    for init_ in zip(n_features, n_targets):
        with pytest.raises(AssertionError):
            lnr = LinearRegressionModel(n_features=init_[0], n_targets=init_[1])
    
def test_linear_regression_model_train_wrong_input_data():
    num_data = 30
    
    # Single feature wrong data input:
    n_features = 2
    n_targets = 1
    lnr = LinearRegressionModel(n_features=n_features, n_targets=n_targets)    
    data = np.random.rand(num_data)
    label = np.random.rand(num_data)
    with pytest.raises(AssertionError):
        lnr.train(data, label)
     
    # Multi-feature wrong data input:
    n_features = 2
    n_targets = 1
    lnr = LinearRegressionModel(n_features=n_features, n_targets=n_targets)    
    data = np.random.rand(num_data, n_features + 1)
    label = np.random.rand(num_data)
    with pytest.raises(AssertionError):
        lnr.train(data, label)
        
    # Single target wrong input label:
    n_features = 2
    n_targets = 2
    lnr = LinearRegressionModel(n_features=n_features, n_targets=n_targets)    
    data = np.random.rand(num_data, n_features)
    label = np.random.rand(num_data)
    with pytest.raises(AssertionError):
        lnr.train(data, label)
        
    # Multi target wrong input label:
    n_features = 2
    n_targets = 3
    lnr = LinearRegressionModel(n_features=n_features, n_targets=n_targets)    
    data = np.random.rand(num_data, n_features)
    label = np.random.rand(num_data, n_targets + 1)
    with pytest.raises(AssertionError):
        lnr.train(data, label)
        
def test_linear_regression_model_train():
    train_data = np.array([[66], [92], [98], [17], [83], [57], [86], [97], [96], [47]])
    train_labels = np.array([125.806, 182.860, 201.525,  60.684, 174.566, 117.808, 174.305, 209.767, 203.265, 90.985])
    test_data = np.array([[84], [47], [61], [48]])
    test_labels = np.array([195.104, 103.436, 121.397,  87.779])
    
    # Single feature, single target:
    n_features = 1
    n_targets = 1
    lnr = LinearRegressionModel(n_features=n_features, n_targets=n_targets)
    train_features = train_data
    train_targets = train_labels
    test_features = test_data
    test_targets = test_labels
    lnr.train(data=train_features, labels=train_targets)
    evaluation = np.array(lnr.evaluate(data=test_features, labels=test_targets))
    model_params = lnr.get_model_params()
    assert(pytest.approx(model_params, 0.001) == np.array([[13.452,  1.903]]))
    assert(pytest.approx(evaluation, 0.001) == np.array([14.407, 0.877]))
    
    # Multi feature, single target: 
    n_features = 2
    n_targets = 1
    lnr = LinearRegressionModel(n_features=n_features, n_targets=n_targets)
    train_features = train_data.reshape(-1, 2)
    train_targets = train_labels[0:len(train_features)]
    test_features = test_data.reshape(-1, 2)
    test_targets = test_labels[0:len(test_features)]
    lnr.train(data=train_features, labels=train_targets)
    evaluation = np.array(lnr.evaluate(data=test_features, labels=test_targets))
    model_params = lnr.get_model_params()
    assert(pytest.approx(model_params, 0.001) == np.array([[510.521,  -2.674,  -2.128]]))
    assert(pytest.approx(evaluation, 0.001) == np.array([100.464,  -3.804]))
    
    # Multi feature, multi target: 
    n_features = 2
    n_targets = 2
    lnr = LinearRegressionModel(n_features=n_features, n_targets=n_targets)
    train_features = train_data.reshape(-1, 2)
    train_targets = train_labels.reshape(-1, 2)
    test_features = test_data.reshape(-1, 2)
    test_targets = test_labels.reshape(-1, 2)
    lnr.train(data=train_features, labels=train_targets)
    evaluation = np.array(lnr.evaluate(data=test_features, labels=test_targets))
    model_params = lnr.get_model_params()
    assert(pytest.approx(model_params, 0.001) == np.array([[-3.078e+01,  2.414e+00, -8.333e-03],\
                                                           [-2.608e+01,  4.217e-01,  1.972e+00]]))
    assert(pytest.approx(evaluation, 0.001) == np.array([12.468,  0.710]))
    
def test_linear_regression_model_set_get_params():
    n_features = 9
    n_targets = 3
    lnr = LinearRegressionModel(n_features=n_features, n_targets=n_targets)
    params = np.random.rand(n_targets, n_features)
    lnr.set_model_params(params)
    
    assert np.array_equal(lnr.get_model_params(), params)


@patch('shfl.model.linear_regression_model.metrics.mean_squared_error')
def test_linear_regression_model_performance(mse_mock):
    n_features = 9
    n_targets = 3
    data = np.random.randint(0, 100, 10).reshape((-1, 1))
    labels = np.random.rand(10)

    lnr = LinearRegressionModel(n_features=n_features, n_targets=n_targets)

    lnr.predict = Mock()
    prediction = labels + 0.5
    lnr.predict.return_value = prediction
    mse_mock.return_value = 4

    lnr._check_data = Mock()
    lnr._check_labels = Mock()

    rmse = lnr.performance(data, labels)

    lnr._check_data.assert_called_once_with(data)
    lnr._check_labels.assert_called_once_with(labels)
    lnr.predict.assert_called_once_with(data)
    mse_mock.assert_called_once_with(labels, prediction)

    assert -rmse == np.sqrt(mse_mock.return_value)
