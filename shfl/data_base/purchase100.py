from tensorflow.python.keras.utils.data_utils import get_file
import numpy as np

from shfl.data_base import data_base as db


class Purchase100(db.DataBase):
    """
    This database loads the \
    [Purchase100 dataset extracted from Kaggle: Acquire Valued Shoppers Challenge](https://www.kaggle.com/c/acquire-valued-shoppers-challenge).
    """
    def load_data(self):
        """
        Load data from Purchase100 dataset

        # Returns
            all_data : train data, train labels, test data and test labels
        """

        path_features = get_file(
            "puchase100_features",
            origin="https://github.com/xehartnort/Purchase100-dataset/releases/download/v1.0/purchase100_features.npy.zip",
            extract=True,
            file_hash= "b0c8c072d80959dfc161f2928aac1c00", # md5 hash
            cache_dir='~/.sherpa-ai')

        path_labels = get_file(
            "puchase100_labels",
            origin="https://github.com/xehartnort/Purchase100-dataset/releases/download/v1.0/purchase100_labels.npy.zip",
            extract=True,
            file_hash= "7b7409c4897f86889dd08a916dd9a111", # md5 hash
            cache_dir='~/.sherpa-ai')

        data = np.load(path_features)
        labels = np.load(path_labels)

        test_size = int(len(data) * 0.1)
        self._train_data, self._train_labels,\
            self._test_data, self._test_labels = db.split_train_test(data, labels, test_size)

        self.shuffle()

        return self.data