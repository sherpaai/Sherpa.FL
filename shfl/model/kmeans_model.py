from shfl.model.model import TrainableModel
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics


class KMeansModel(TrainableModel):
    """
    This class offers support for scikit-learn K-Means model. It implements [TrainableModel](../model/#trainablemodel-class)

    # Arguments:
        n_clusters: number of clusters.
        init: Method of initialization. {‘k-means++’, ‘random’, ndarray}, default=’k-means++’.
            If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
            When ‘random’: choose n_clusters observations  (rows) at random from data for the initial centroids.
        n_init: Number of time the k-means algorithm will be run with different centroid seeds (default 10).
    """

    def __init__(self, n_clusters, n_features, init='k-means++', n_init=10):
        self._model = KMeans(n_clusters=n_clusters, init=init, n_init=n_init)
        self._init = init
        self._n_features = n_features
        self._n_init = n_init
        self._model._n_threads = None # compatibility: should be removed from scikit-learn

        if type(init) is np.ndarray:
            self._model.cluster_centers_ = init
        else:
            self._model.cluster_centers_ = np.zeros((n_clusters, n_features))

    def train(self, data, labels=None):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        # Arguments
            data: Data, array-like of shape (n_samples, n_features)
            labels: None.
        """
        self._model.fit(data)

    def predict(self, data):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        # Arguments:
            data: Data, array-like of shape (n_samples, n_features)

        # Returns:
            predicted_labels: array with prediction labels for data argument.
        """
        predicted_labels = self._model.predict(data)
        return predicted_labels

    def evaluate(self, data, labels):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)
        Metrics for evaluating model's performance.

        # Arguments:
            data: Data, array-like of shape (n_samples, n_features)
            labels: Target, array-like of shape (n_samples,) or (n_samples, n_targets)

        # Returns:
            homo: Homogeneity score (see [Homogeneity metric] /
            (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html))
            compl: Completeness score (see [Completeness metric] /
            (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html))
            v_means: v-measure score (see [V-measure cluster] /
            (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html))
            rai: Adjusted rand score (see [Rand index adjusted metric] /
            (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html))
        """
        prediction = self.predict(data)

        homo = metrics.homogeneity_score(labels, prediction)
        compl = metrics.completeness_score(labels, prediction)
        v_meas = metrics.v_measure_score(labels, prediction)
        rai = metrics.adjusted_rand_score(labels, prediction)
        return homo, compl, v_meas, rai

    def performance(self, data, labels):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        # Arguments:
            data: Data, array-like of shape (n_samples, n_features)
            labels: Target, array-like of shape (n_samples,) or (n_samples, n_targets)

        # Returns:
            v_means: v-measure score (see [V-measure cluster] /
            (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html))
        """
        prediction = self.predict(data)
        v_meas = metrics.v_measure_score(labels, prediction)

        return v_meas

    def get_model_params(self):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        # Returns:
            centers: array with cluster centers kmeans model.
        """
        return self._model.cluster_centers_

    def set_model_params(self, params):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        # Arguments:
            params: representation of model params to assign
        """
        if np.array_equal(params, np.zeros((params.shape[0], params.shape[1]))):
            self.__init__(n_clusters=params.shape[0], n_features=self._n_features, init=self._init,
                          n_init=self._n_init)
        else:
            self.__init__(n_clusters=params.shape[0], n_features=self._n_features, init=params, n_init=1)
