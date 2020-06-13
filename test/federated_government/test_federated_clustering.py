from shfl.federated_government.federated_clustering import FederatedClustering, ClusteringDataBases
from shfl.federated_aggregator.cluster_fedavg_aggregator import ClusterFedAvgAggregator
from shfl.model.kmeans_model import KMeansModel
from unittest.mock import Mock, patch

import numpy as np


def test_FederatedClustering():
    database = 'IRIS'
    cfg = FederatedClustering(database, iid=True, num_nodes=3, percent=20)

    module = ClusteringDataBases.__members__[database].value
    data_base = module()
    train_data, train_labels, test_data, test_labels = data_base.load_data()

    assert cfg._test_data is not None
    assert cfg._test_labels is not None
    assert cfg._num_clusters == len(np.unique(train_labels))
    assert cfg._num_features == train_data.shape[1]
    assert isinstance(cfg._aggregator, ClusterFedAvgAggregator)
    assert isinstance(cfg._model, KMeansModel)
    assert cfg._federated_data is not None

    cfg = FederatedClustering(database, iid=False, num_nodes=3, percent=20)

    assert cfg._test_data is not None
    assert cfg._test_labels is not None
    assert cfg._num_clusters == len(np.unique(train_labels))
    assert cfg._num_features == train_data.shape[1]
    assert isinstance(cfg._aggregator, ClusterFedAvgAggregator)
    assert isinstance(cfg._model, KMeansModel)
    assert cfg._federated_data is not None

def test_FederatedClustering_wrong_database():
    cfg = FederatedClustering('MNIST', iid=True, num_nodes=3, percent=20)

    assert cfg._test_data is None


def test_run_rounds():
    cfg = FederatedClustering('IRIS', iid=True, num_nodes=3, percent=20)

    cfg.deploy_central_model = Mock()
    cfg.train_all_clients = Mock()
    cfg.evaluate_clients = Mock()
    cfg.aggregate_weights = Mock()
    cfg.evaluate_global_model = Mock()

    cfg.run_rounds(1)

    cfg.deploy_central_model.assert_called_once()
    cfg.train_all_clients.assert_called_once()
    cfg.evaluate_clients.assert_called_once_with(cfg._test_data, cfg._test_labels)
    cfg.aggregate_weights.assert_called_once()
    cfg.evaluate_global_model.assert_called_once_with(cfg._test_data, cfg._test_labels)


def test_run_rounds_wrong_database():
    cfg = FederatedClustering('EMNIST', iid=True, num_nodes=3, percent=20)

    cfg.deploy_central_model = Mock()
    cfg.train_all_clients = Mock()
    cfg.evaluate_clients = Mock()
    cfg.aggregate_weights = Mock()
    cfg.evaluate_global_model = Mock()

    cfg.run_rounds(1)

    cfg.deploy_central_model.assert_not_called()
    cfg.train_all_clients.assert_not_called()
    cfg.evaluate_clients.assert_not_called()
    cfg.aggregate_weights.assert_not_called()
    cfg.evaluate_global_model.assert_not_called()


@patch('shfl.federated_government.federated_clustering.KMeansModel')
def test_model_builder(mock_kmeans):
    cfg = FederatedClustering('IRIS', iid=True, num_nodes=3, percent=20)

    model = cfg.model_builder()

    assert isinstance(model, Mock)
    mock_kmeans.assert_called_with(n_clusters=cfg._num_clusters, n_features=cfg._num_features)