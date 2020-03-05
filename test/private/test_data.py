import numpy as np
import pytest

from shfl.private.data import LabeledData
from shfl.private.data import DataAccessDefinition


def test_labeled_data():
    data = np.random.rand(10)
    label = np.random.rand(1)
    labeled_data = LabeledData(data, label)
    for i in range(len(data)):
        assert labeled_data.data[i] == data[i]
    assert labeled_data.label == label
    new_data = np.random.rand(10)
    labeled_data.data = new_data
    for i in range(len(new_data)):
        assert labeled_data.data[i] == new_data[i]
    new_label = np.random.rand(1)
    labeled_data.label = new_label
    assert labeled_data.label == new_label


def test_data_access_definition_bad_contruction():
    with pytest.raises(ValueError):
        DataAccessDefinition()
