from shfl.model.model import TrainableModel
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import copy


class DeepLearningModelPyTorch(TrainableModel):
    """
    This class offers support for PyTorch models. It implements [TrainableModel](../model/#trainablemodel-class)

    # Arguments:
        model: Compiled model, ready to train
        criterion: Loss function to apply
        optimizer: Optimizer to apply
        batch_size: batch_size to apply
        epochs: Number of epochs
        metrics: Metrics for apply. Dictionary {name: function to apply, ...}. Default shows loss and accuracy
        device: Device where it will run. Default cpu
    """
    def __init__(self, model, criterion, optimizer, batch_size=32, epochs=1, metrics=None, device="cpu"):
        self._model = model
        self._data_shape = model[0].in_channels
        self._labels_shape = model[-2].out_features
        self._criterion = criterion
        self._optimizer = optimizer

        self._batch_size = batch_size
        self._epochs = epochs
        self._device = device
        self._metrics = metrics

    def train(self, data, labels):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        # Arguments
            data: Data with shape NxD (N: Number of elements; D: Dimensions)
            labels: Labels for data with One Hot Encoded format.
        """
        self._check_data(data)
        self._check_labels(labels)

        dataset = TensorDataset(torch.from_numpy(data), torch.from_numpy(labels))
        trainloader = DataLoader(dataset, self._batch_size, False)

        self._model.to(self._device)
        for t in range(self._epochs):
            for element in trainloader:
                inputs, y_true = element[0].to(self._device), element[1].to(self._device)

                self._optimizer.zero_grad()

                y_pred = self._model(inputs)

                loss = self._criterion(y_pred, torch.argmax(y_true, -1))

                self._model.zero_grad()

                loss.backward()
                self._optimizer.step()

    def predict(self, data):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        # Arguments:
            data: Data with shape NxD (N: Number of elements; D: Dimensions)

        # Returns:
            predictions: Predictions for data argument
        """
        self._check_data(data)

        dataset = TensorDataset(torch.from_numpy(data))
        dataloader = DataLoader(dataset, self._batch_size, False)

        y_pred = []
        self._model.to(self._device)
        with torch.no_grad():
            for element in dataloader:
                inputs = element[0].to(self._device)

                batch_y_pred = torch.argmax(self._model(inputs), -1)

                y_pred.extend(batch_y_pred.cpu().numpy())

        return np.array(y_pred)

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

        dataset = TensorDataset(torch.from_numpy(data), torch.from_numpy(labels))
        dataloader = DataLoader(dataset, self._batch_size, False)

        self._model.to(self._device)
        all_y_pred = []
        with torch.no_grad():
            for element in dataloader:
                inputs, y_true = element[0].to(self._device), element[1].to(self._device)

                y_pred = self._model(inputs)

                all_y_pred.extend(y_pred.cpu().numpy())

            all_y_pred = torch.from_numpy(np.array(all_y_pred))
            labels_t = torch.from_numpy(labels)
            val_loss = self._criterion(all_y_pred, torch.argmax(labels_t, -1))

            correct_predict = (torch.argmax(all_y_pred, -1).numpy() == torch.argmax(labels_t, -1).numpy()).sum()
            val_acc = correct_predict / len(labels)

            metrics = [val_loss.item(), val_acc]
            if self._metrics is not None:
                for name, metric in self._metrics.items():
                    metrics.append(metric(all_y_pred.cpu().numpy(), labels))

        return metrics

    def performance(self, data, labels):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        # Arguments:
            data: Data with shape NxD (N: Number of elements; D: Dimensions)
            labels: Labels for data with One Hot Encoded format.

        # Returns:
            metric: Returns the value of the main metric.
        """
        return self.evaluate(data, labels)[0]

    def get_model_params(self):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        # Returns
            weights: Returns the model weights.
        """
        weights = []
        for param in self._model.parameters():
            weights.append(param.cpu().data.numpy())

        return weights

    def set_model_params(self, params):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        # Arguments:
            params: array with the model weights
        """
        with torch.no_grad():
            for ant, post in zip(self._model.parameters(), params):
                ant.data = torch.from_numpy(post)

    def _check_data(self, data):
        """
        Method that checks if the data dimension if correct.
        """
        if data.shape[1] != self._data_shape:
            raise AssertionError("Data need to have the same dimension described by the model " + str(self._data_shape) +
                                 " .Current data has dimension " + str(data.shape[1]))

    def _check_labels(self, labels):
        """
        Method that checks if the labels dimension if correct.
        """
        if labels.shape[-1] != self._labels_shape:
            raise AssertionError("Labels need to have the same shape described by the model " + str(self._labels_shape)
                                 + " .Current data has shape " + str(labels.shape[1:]))
