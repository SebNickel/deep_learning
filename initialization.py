from typing import Tuple
import numpy
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
