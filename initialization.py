from typing import Callable, Tuple
import numpy
from numpy import ndarray
import theano

def normalization_factor_for_tanh(input_dim: int,
                                  linear_output_dim: int) -> float:

    return numpy.sqrt(6.0 / (input_dim + linear_output_dim))


def normalization_factor_for_sigmoid(input_dim: int,
                                     linear_output_dim: int) -> float:

    return 4 * normalization_factor_for_tanh(input_dim, linear_output_dim)


def zero_initialization() -> Callable[[Tuple], ndarray]:

    return lambda shape: numpy.zeros(shape=shape, dtype=theano.config.floatX)


def uniform_initialization(normalization_factor: float) -> Callable[[Tuple], ndarray]:

    return lambda shape: numpy.random.uniform(-normalization_factor, normalization_factor, shape)