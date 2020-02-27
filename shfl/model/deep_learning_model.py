from keras.callbacks import EarlyStopping
from shfl.model.model import TrainableModel


class DeepLearningModel(TrainableModel):

    def __init__(self, model, batch_size=None, epochs=1):
        self._model = model
        self._data_shape = model.layers[0].get_input_shape_at(0)[1:]
        self._label_shape = model.layers[-1].get_output_shape_at(0)[1:]

        self._batch_size = batch_size
        self._epochs = epochs

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
