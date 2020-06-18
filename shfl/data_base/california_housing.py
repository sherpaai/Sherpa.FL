import sklearn.datasets
from shfl.data_base import data_base as db


class CaliforniaHousing(db.DataBase):
    """
    This database loads the \
    [California housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html#sklearn.datasets.fetch_california_housing)
    from sklearn, mainly for regression tasks.
    """
    def load_data(self):
        """
        Load data from california housing package

        # Returns
            all_data : train data, train labels, test data and test labels
        """

        all_data = sklearn.datasets.fetch_california_housing()
        data = all_data["data"]
        labels = all_data["target"]

        test_size = int(len(data) * 0.1)
        self._train_data, self._train_labels,\
            self._test_data, self._test_labels = db.split_train_test(data, labels, test_size)

        self.shuffle()

        return self.data
