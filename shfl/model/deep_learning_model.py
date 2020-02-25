from abc import ABC
from keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np

from shfl.model.model import TrainableModel


class DeepLearningModel(TrainableModel, ABC):

    def __init__(self, model, batch_size=None, epochs=1):
        self._model = model
        self._data_shape = model.layers[0].input_shape[1:]
        self._label_shape = model.layers[-1].output_shape[1:]

        self._batch_size = batch_size
        self._epochs = epochs


class KerasDeepLearningModel(DeepLearningModel):

    def train(self, data, label):
        if data.shape[1:] != self._data_shape:
            raise AssertionError("Data need to have the same shape described by the model " + str(self._data_shape) +
                                 " .Current data has shape " + str(data.shape[1:]))

        if label.shape[1:] != self._label_shape:
            raise AssertionError("Label need to have the same shape described by the model " + str(self._label_shape) +
                                 " .Current data has shape " + str(label.shape[1:]))

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
        self._model.fit(x=data, y=label, batch_size=self._batch_size, epochs=self._epochs, validation_split=0.2,
                        verbose=0, shuffle=False, callbacks=[early_stopping])

    def predict(self, data):
        if data.shape[1:] != self._data_shape:
            raise AssertionError("Data need to have the same shape described by the model" + self._data_shape)

        return self._model.predict(data, batch_size=self._batch_size).argmax(axis=-1)

    def get_model_params(self):
        return self._model.get_weights()

    def set_model_params(self, params):
        self._model.set_weights(params)


class TensorflowDeepLearningModel(DeepLearningModel):


    def __init__(self, model, batch_size=None, epochs=1):
        super(DeepLearningModel, self).__init__(model, batch_size, epochs)

        self._optimizers = {
            'adadelta': tf.optimizers.Adadelta(),
            'adagrad': tf.optimizers.Adagrad(),
            'adam': tf.optimizers.Adam(),
            'adamax': tf.optimizers.Adamax(),
            'ftrl': tf.optimizers.Ftrl(),
            'nadam': tf.optimizers.Nadam(),
            'rmsprop': tf.optimizers.RMSprop(),
            'sgd': tf.optimizers.SGD()
        }
        self._loss_fns = {
            'binary_crossentropy': tf.losses.BinaryCrossentropy(),
            'categorical_crossentropy': tf.losses.CategoricalCrossentropy(),
            'categorical_hinge': tf.losses.CategoricalHinge(),
            'cosine_similarity': tf.losses.CosineSimilarity(),
            'hinge': tf.losses.Hinge(),
            'huber': tf.losses.Huber(),
            'kl_divergence': tf.losses.KLDivergence(),
            'log_cosh': tf.losses.LogCosh(),
            'mean_absolute_error': tf.losses.MeanAbsoluteError(),
            'mean_absolute_percentage_error': tf.losses.MeanAbsolutePercentageError(),
            'mean_squared_error': tf.losses.MeanSquaredError(),
            'mean_squared_logarithmic_error': tf.losses.MeanSquaredLogarithmicError(),
            'poisson': tf.losses.Poisson(),
            'reduction': tf.losses.Reduction(),
            'sparse_categorical_crossentropy': tf.losses.SparseCategoricalCrossentropy(),
            'squared_hinge': tf.losses.SquaredHinge()
        }

    @tf.function
    def train_on_batch(self, x, y):
        """
        Method for train on batch

        Parameters
        ----------
        x: matrix
            Train data
        y: array
            Labels data

        Return
        ------
        loss : float
            Loss value from a batch
        """
        with tf.GradientTape() as tape:
            predictions = self._model(x, training=True)
            loss = self._loss_fn(y, predictions)

        # Obtain the gradient
        gradient = tape.gradient(loss, self._model.trainable_variables)
        # Apply the gradient
        self._optimizer.apply_gradients(zip(gradient, self._model.trainable_variables))

        # Update metrics
        self._acc_metric(y, predictions)
        self._loss_metric(loss)

        return loss


    def train(self, data, label):
        if data.shape[1:] != self._data_shape:
            raise AssertionError("Data need to have the same shape described by the model " + str(self._data_shape) +
                                 " .Current data has shape " + str(data.shape[1:]))

        if label.shape[1:] != self._label_shape:
            raise AssertionError("Label need to have the same shape described by the model " + str(self._label_shape) +
                                 " .Current data has shape " + str(label.shape[1:]))

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
        self._model.fit(x=data, y=label, batch_size=self._batch_size, epochs=self._epochs, validation_split=0.2,
                        verbose=0, shuffle=False, callbacks=[early_stopping])

    def predict(self, data):
        """
        Predict labels for data using the trained model

        Parameters
        ----------
        dataset: matrix
            data to classify
        batch_size: int
            Batch size used for prediction (default is None)

        Return
        ------
        predictions : matrix
            predictions for dataset
        loss : float
            Loss for a prediction
        """

        predictions = []
        dataset = data.batch(self._batch_size)
        for features, labels in dataset:
            pred = self._model(features)
            predictions.append(pred.numpy())

        return np.vstack(predictions)

