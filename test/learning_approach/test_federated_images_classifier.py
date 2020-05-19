from shfl.learning_approach.federated_images_classifier import FederatedImagesClassifier, ImagesDataBases
from shfl.model.deep_learning_model import DeepLearningModel
from shfl.federated_aggregator.federated_aggregator import FederatedAggregator
from shfl.private.federated_operation import FederatedData
from unittest.mock import Mock
import pytest
import random
import string


def test_images_classifier_iid():
    example_database = list(ImagesDataBases.__members__.keys())[0]
    fic = FederatedImagesClassifier(example_database)

    for node in fic._federated_data:
        assert isinstance(node._model, DeepLearningModel)

    assert isinstance(fic._model, DeepLearningModel)
    assert isinstance(fic._aggregator, FederatedAggregator)
    assert isinstance(fic._federated_data, FederatedData)

    assert fic._test_data is not None
    assert fic._test_labels is not None


def test_images_classifier_noiid():
    example_database = list(ImagesDataBases.__members__.keys())[0]
    fic = FederatedImagesClassifier(example_database, False)

    for node in fic._federated_data:
        assert isinstance(node._model, DeepLearningModel)

    assert isinstance(fic._model, DeepLearningModel)
    assert isinstance(fic._aggregator, FederatedAggregator)
    assert isinstance(fic._federated_data, FederatedData)

    assert fic._test_data is not None
    assert fic._test_labels is not None


def test_images_classifier_wrong_database():
    letters = string.ascii_lowercase
    wrong_database = ''.join(random.choice(letters) for i in range(10))

    fic = FederatedImagesClassifier(wrong_database)

    assert fic._test_data == None
    with pytest.raises(AttributeError):
        fic._model
        fic._aggregator
        fic._federated_data


def test_run_rounds():
    example_database = list(ImagesDataBases.__members__.keys())[0]

    fic = FederatedImagesClassifier(example_database)

    fic.deploy_central_model = Mock()
    fic.train_all_clients = Mock()
    fic.evaluate_clients = Mock()
    fic.aggregate_weights = Mock()
    fic.evaluate_global_model = Mock()

    fic.run_rounds(1)

    fic.deploy_central_model.assert_called_once()
    fic.train_all_clients.assert_called_once()
    fic.evaluate_clients.assert_called_once_with(fic._test_data, fic._test_labels)
    fic.aggregate_weights.assert_called_once()
    fic.evaluate_global_model.assert_called_once_with(fic._test_data, fic._test_labels)


def test_run_rounds_wrong_database():
    letters = string.ascii_lowercase
    wrong_database = ''.join(random.choice(letters) for i in range(10))

    fic = FederatedImagesClassifier(wrong_database)

    fic.deploy_central_model = Mock()
    fic.train_all_clients = Mock()
    fic.evaluate_clients = Mock()
    fic.aggregate_weights = Mock()
    fic.evaluate_global_model = Mock()

    fic.run_rounds(1)

    fic.deploy_central_model.assert_not_called()
    fic.train_all_clients.assert_not_called()
    fic.evaluate_clients.assert_not_called()
    fic.aggregate_weights.assert_not_called()
    fic.evaluate_global_model.assert_not_called()
