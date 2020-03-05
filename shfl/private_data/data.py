from shfl.private_data.query import IdentityFunction
from shfl.differential_privacy.dp_mechanism import UnrandomizedMechanism


class LabeledData:
    """
    Class to represent labeled data
    """
    def __init__(self, data, label):
        self._data = data
        self._label = label

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label


class DataAccessDefinition:
    """
    Class to represent how private data can be accessed
    """
    def __init__(self, query=None, dp_mechanism=None):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            query: Description of `param1`.
            dp_mechanism: Description of `param2`. Multiple
                lines are supported.
        """
        if query is None and dp_mechanism is None:
            raise ValueError("You can't define a data access without setting one of query or dp_mechanism")

        if query is None:
            self._query = IdentityFunction()
        else:
            self._query = query

        if dp_mechanism is None:
            self._dp_mechanism = UnrandomizedMechanism()
        else:
            self._dp_mechanism = dp_mechanism

    @property
    def query(self):
        return self._query

    @property
    def dp_mechanism(self):
        return self._dp_mechanism


class UnprotectedAccess(DataAccessDefinition):
    """
    This class implements access to data without security
    """
    def __init__(self):
        super().__init__(IdentityFunction())
