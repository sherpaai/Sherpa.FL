from sklearn.datasets import fetch_lfw_people
from shfl.data_base import data_base as db


class Lfw(db.DataBase):
    """
    This database loads the \
    [Labeled faces in the wild dataset](https://scikit-learn.org/stable/datasets/index.html#labeled-faces-in-the-wild-dataset)
    from sklearn, mainly for face recognition task.
    """

    def load_data(self):
        """
        Load data from lfw package

        # Returns
            all_data : train data, train labels, test data and test labels
        """
        all_data = fetch_lfw_people()
        data = all_data["images"]
        labels = all_data["target"]

        test_size = int(len(data) * 0.1)
        self._train_data, self._train_labels,\
            self._test_data, self._test_labels = db.split_train_test(
                data, labels, test_size)

        self.shuffle()

        return self.data
