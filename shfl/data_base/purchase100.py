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
            "purchase100",
            origin="https://github.com/xehartnort/Purchase100-dataset/releases/download/v1.1/purchase100.npz",
            extract=True,
            file_hash= "0d7538b9806e7ee622e1a252585e7768", # md5 hash
            cache_dir='~/.sherpa-ai')

        all_data = np.load(path_features)
        data = all_data['features']
        labels = all_data['labels']

        test_size = int(len(data) * 0.1)
        self._train_data, self._train_labels,\
            self._test_data, self._test_labels = db.split_train_test(data, labels, test_size)

        self.shuffle()

        return self.data