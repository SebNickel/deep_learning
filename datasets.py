import numpy
from numpy import ndarray
import theano
from theano import tensor as T
import pickle


class SharedDataset:

    def __init__(self,
                 vectors: ndarray,
                 labels: ndarray):

        self.x = theano.shared(
            value=numpy.asarray(
                vectors,
                dtype=theano.config.floatX
            ),
            name='x',
            borrow=True
        )

        # Intermittent conversion to float for GPU-compatibility.
        y_as_floats = theano.shared(
            value=numpy.asarray(
                labels,
                dtype=theano.config.floatX
            ),
            name='y',
            borrow=True
        )

        self.y = T.cast(y_as_floats, 'int32')

    @property
    def vectors(self):

        return self.x.get_value(borrow=True)

    @property
    def labels(self):

        return self.y.get_value(borrow=True)

    @property
    def size(self):

        return self.vectors.shape[0]


def save(dataset: SharedDataset,
         file_path: str):

    with open(file_path, 'wb') as file:

        pickle._dump(dataset, file)


def load(file_path: str) -> SharedDataset:

    with open(file_path, 'rb') as file:

        return pickle.load(file)
