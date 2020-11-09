from tensorflow.keras.callbacks import EarlyStopping
from shfl.model.model import TrainableModel
import tensorflow as tf
import copy


class DeepLearningModel(TrainableModel):
    """
    This class offers support for Keras and tensorflow models. It implements [TrainableModel](../model/#trainablemodel-class)

    # Arguments:
        model: Compiled model, ready to train
        criterion: Loss function to apply
        optimizer: Optimizer to apply
        batch_size: batch_size to apply
        epochs: Number of epochs
        metrics: Metrics for apply. List of tensorflow metrics.
    """
    def __init__(self, model, criterion, optimizer, batch_size=None, epochs=1, metrics=None):
        self._model = model
        self._data_shape = model.layers[0].get_input_shape_at(0)[1:]
        self._labels_shape = model.layers[-1].get_output_shape_at(0)[1:]

        self._batch_size = batch_size
        self._epochs = epochs
        self._criterion = criterion
        self._optimizer = optimizer
        self._metrics = metrics

        self._model.compile(optimizer=self._optimizer, loss=self._criterion, metrics=self._metrics)

    def train(self, data, labels):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        # Arguments
            data: Data with shape NxD (N: Number of elements; D: Dimensions)
            labels: Labels for data with One Hot Encoded format.
        """
        self._check_data(data)
        self._check_labels(labels)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
        self._model.fit(x=data, y=labels, batch_size=self._batch_size, epochs=self._epochs, validation_split=0.2,
                        verbose=0, shuffle=False, callbacks=[early_stopping])

    def predict(self, data):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        # Arguments:
            data: Data with shape NxD (N: Number of elements; D: Dimensions)

        # Returns:
            predictions: Predictions for data argument
        """
        self._check_data(data)

        return self._model.predict(data, batch_size=self._batch_size).argmax(axis=-1)

    def evaluate(self, data, labels):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        # Arguments:
            data: Data with shape NxD (N: Number of elements; D: Dimensions)
            labels: Labels for data with One Hot Encoded format.

        # Returns:
            metrics: Returns metrics for data argument
        """
        self._check_data(data)
        self._check_labels(labels)

        return self._model.evaluate(data, labels, verbose=0)

    def performance(self, data, labels):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        # Arguments:
            data: Data with shape NxD (N: Number of elements; D: Dimensions)
            labels: Labels for data with One Hot Encoded format.

        # Returns:
            metric: Returns the value of the main metric.
        """
        self._check_data(data)
        self._check_labels(labels)

        return self._model.evaluate(data, labels, verbose=0)[0]

    def get_model_params(self):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        # Returns
            weights: Returns the model weights.
        """
        return self._model.get_weights()

    def set_model_params(self, params):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        # Arguments:
            params: array with the model weights
        """
        self._model.set_weights(params)

    def _check_data(self, data):
        """
        Method that checks if the data dimension if correct.
        """
        if data.shape[1:] != self._data_shape:
            raise AssertionError("Data need to have the same shape described by the model " + str(self._data_shape) +
                                 " .Current data has shape " + str(data.shape[1:]))

    def _check_labels(self, labels):
        """
        Method that checks if the labels dimension if correct.
        """
        if labels.shape[1:] != self._labels_shape:
            raise AssertionError("Labels need to have the same shape described by the model " + str(self._labels_shape)
                                 + " .Current data has shape " + str(labels.shape[1:]))

    def __deepcopy__(self, memo):
        """
        Overwrite deepcopy method
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_model":
                model = tf.keras.models.clone_model(v)
                model.set_weights(v.get_weights())
                setattr(result, k, model)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        result._model.compile(optimizer=result._optimizer, loss=result._criterion, metrics=result._metrics)
        return result
