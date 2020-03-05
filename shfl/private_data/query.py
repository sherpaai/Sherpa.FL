import numpy as np
import abc


class Query(abc.ABC):
    """
    This class represents a query over private data
    """

    @abc.abstractmethod
    def get(self, data):
        """

        Parameters
        ----------
        data: object
            Data to process

        Returns
        -------
        answer: object
            Result of apply query over data
        """


class IdentityFunction(Query):
    """Reshapes an output to a certain shape.

    # Arguments
        target_shape: target shape. Tuple of integers.
            Does not include the batch axis.

    # Input shape
        Arbitrary, although all dimensions in the input shaped must be fixed.
        Use the keyword argument `input_shape`
        (tuple of integers, does not include the batch axis)
        when using this layer as the first layer in a model.

    # Output shape
        `(batch_size,) + target_shape`

    # Example

    ```python
        # as first layer in a Sequential model
        model = Sequential()
        model.add(Reshape((3, 4), input_shape=(12,)))
        # now: model.output_shape == (None, 3, 4)
        # note: `None` is the batch dimension
        # as intermediate layer in a Sequential model
        model.add(Reshape((6, 2)))
        # now: model.output_shape == (None, 6, 2)
        # also supports shape inference using `-1` as dimension
        model.add(Reshape((-1, 2, 2)))
        # now: model.output_shape == (None, 3, 2, 2)
    ```
    """
    def get(self, data):
        return data


class Mean(Query):
    """
    Implements mean over data array
    """
    def get(self, data):
        return np.mean(data)
