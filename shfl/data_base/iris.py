import sklearn.datasets
from shfl.data_base import data_base as db


class Iris(db.DataBase):
    """
    This database loads the \
    [Irisdataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
    from sklearn, mainly for clustering tasks.
    """
    def load_data(self):
        """
        Load data from iris package

        # Returns
            all_data : train data, train labels, test data and test labels
        """
        all_data = sklearn.datasets.load_iris()
        data = all_data["data"]
        labels = all_data["target"]

        test_size = int(len(data) * 0.1)
        self._train_data, self._train_labels,\
            self._test_data, self._test_labels = db.split_train_test(data, labels, test_size)

        self.shuffle()

        return self.data
