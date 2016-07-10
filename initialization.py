from typing import Tuple, Callable
import numpy
from numpy import ndarray
import theano
from theano import tensor as T


def zero_initialize(shape: Tuple,
                    name: str=None,
                    dtype=theano.config.floatX) -> T.TensorVariable:

    return theano.shared(
        value=numpy.zeros(
            shape=shape,
            dtype=dtype
        ),
        name=name,
        borrow=True
    )


def randomly_initialize(shape: Tuple,
                        distribution: Callable[[Tuple], ndarray],
                        name: str=None) -> T.TensorVariable:

    random_array = distribution(shape)

    return theano.shared(
        value=random_array,
        name=name,
        borrow=True
    )