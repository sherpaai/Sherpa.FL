import numpy as np
import pytest

from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from shfl.model.linear_classifier_model import LinearClassifierModel


def test_linear_classifier_model_initialization_binary_classes():
    n_features = 9
    classes = ['a', 'b']
    lgr = LinearClassifierModel(n_features=n_features, classes=classes)
    n_classes = 1 # Binary classification
    assert np.shape(lgr._model.intercept_) == (n_classes,)
    assert np.shape(lgr._model.coef_) == (n_classes, n_features)
    assert np.shape(lgr.get_model_params()) == (n_classes, n_features + 1)
    assert np.array_equal(classes, lgr._model.classes_)


def test_linear_classifier_model_initialization_multiple_classes():
    n_features = 9
    classes = ['a', 'b', 'c']
    lgr = LinearClassifierModel(n_features=n_features, classes=classes)
    n_classes = len(classes)
    assert np.shape(lgr._model.intercept_) == (n_classes,)
    assert np.shape(lgr._model.coef_) == (n_classes, n_features)
    assert np.shape(lgr.get_model_params()) == (n_classes, n_features + 1)
    assert np.array_equal(classes, lgr._model.classes_)

    
def test_linear_classifier_model_wrong_initialization():
    n_features = [9.5, -1, 9, 9] 
    classes = [['a', 'b', 'c'],
               ['a', 'b', 'c'],
               ['b'],
               ['a', 'b', 'a']]
    for init_ in zip(n_features, classes):
        with pytest.raises(AssertionError):
            lgr = LinearClassifierModel(n_features=init_[0], classes=init_[1])
            
            
def test_linear_classifier_model_train_wrong_input_data():
    num_data = 30
    
    # Single feature wrong data input:
    n_features = 2
    classes = ['a', 'b']
    lgr = LinearClassifierModel(n_features=n_features, classes=classes)
    data = np.random.rand(num_data, )
    label = np.random.choice(a=classes, size=num_data, replace=True)
    with pytest.raises(AssertionError):
        lgr.train(data, label)
     
    # Multi-feature wrong data input:
    n_features = 2
    classes = ['a', 'b']
    lgr = LinearClassifierModel(n_features=n_features, classes=classes)
    data = np.random.rand(num_data, n_features + 1)
    label = np.random.choice(a=classes, size=num_data, replace=True)
    with pytest.raises(AssertionError):
        lgr.train(data, label)
        
    # Wrong classes input label on train and predict:
    n_features = 2
    classes = ['a', 'b']
    lgr = LinearClassifierModel(n_features=n_features, classes=classes)
    label = np.random.choice(a=classes, size=num_data, replace=True)
    label[0] = 'c'
    with pytest.raises(AssertionError):
        lgr._check_labels_train(label)
    with pytest.raises(AssertionError):
        lgr._check_labels_predict(label)
      
    
def test_linear_classifier_model_set_get_params():
    n_features = 9
    classes = ['a', 'b', 'c']
    lgr = LinearClassifierModel(n_features=n_features, classes=classes)
    params = np.random.rand(len(classes), n_features)
    lgr.set_model_params(params)
    
    assert np.array_equal(lgr.get_model_params(), params)
           

def test_logistic_regression_model_train_evaluate():
    data, labels = load_iris(return_X_y=True)
    randomize = np.arange(len(labels))
    np.random.shuffle(randomize)
    data = data[randomize, ]
    labels = labels[randomize]
    dim = 100
    train_data = data[0:dim, ]
    train_labels = labels[0:dim]
    test_data = data[dim:, ]
    test_labels = labels[dim:]

    model = LogisticRegression(max_iter=150)
    lgr = LinearClassifierModel(n_features=np.shape(train_data)[1], classes=np.unique(train_labels), model=model)
    lgr.train(data=train_data, labels=train_labels)
    evaluation = np.array(lgr.evaluate(data=test_data, labels=test_labels))
    performance = lgr.performance(data=test_data, labels=test_labels)
    prediction = lgr.predict(data=test_data)
    model_params = lgr.get_model_params()
    
    lgr_ref = LogisticRegression(max_iter=150).fit(train_data, train_labels)
    prediction_ref = lgr_ref.predict(test_data)
    
    assert np.array_equal(model_params, np.column_stack((lgr_ref.intercept_, lgr_ref.coef_)))
    assert np.array_equal(prediction, prediction_ref)
    assert np.array_equal(evaluation, np.array((metrics.balanced_accuracy_score(test_labels, prediction_ref),\
                                               metrics.cohen_kappa_score(test_labels, prediction_ref))))
    assert performance == metrics.balanced_accuracy_score(test_labels, prediction_ref)
    
    
def test_linearSVC_model_train_evaluate():
    data, labels = load_iris(return_X_y=True)
    randomize = np.arange(len(labels))
    np.random.shuffle(randomize)
    data = data[randomize, ]
    labels = labels[randomize]
    dim = 100
    train_data = data[0:dim, ]
    train_labels = labels[0:dim]
    test_data = data[dim:, ]
    test_labels = labels[dim:]

    model = LinearSVC(random_state=123)
    svc = LinearClassifierModel(n_features=np.shape(train_data)[1], classes=np.unique(train_labels), model=model)
    svc.train(data=train_data, labels=train_labels)
    evaluation = np.array(svc.evaluate(data=test_data, labels=test_labels))
    performance = svc.performance(data=test_data, labels=test_labels)
    prediction = svc.predict(data=test_data)
    model_params = svc.get_model_params()
    
    svc_ref = LinearSVC(random_state=123).fit(train_data, train_labels)
    prediction_ref = svc_ref.predict(test_data)
    
    assert np.array_equal(model_params, np.column_stack((svc_ref.intercept_, svc_ref.coef_)))
    assert np.array_equal(prediction, prediction_ref)
    assert np.array_equal(evaluation, np.array((metrics.balanced_accuracy_score(test_labels, prediction_ref),\
                                               metrics.cohen_kappa_score(test_labels, prediction_ref))))
    assert performance == metrics.balanced_accuracy_score(test_labels, prediction_ref)