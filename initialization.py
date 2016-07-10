from typing import Callable, Tuple
import numpy
from numpy import ndarray
import theano

def normalization_factor_for_tanh(num_units_in_previous_layer: int,
                                  num_units_in_this_layer: int) -> float:

    return numpy.sqrt(6.0 / (num_units_in_previous_layer + num_units_in_this_layer))


def normalization_factor_for_sigmoid(num_units_in_previous_layer: int,
                                     num_units_in_this_layer: int) -> float:

    return 4 * normalization_factor_for_tanh(num_units_in_previous_layer, num_units_in_this_layer)


def zero_initialization() -> Callable[[Tuple], ndarray]:

    return lambda shape: numpy.zeros(shape=shape, dtype=theano.config.floatX)


def uniform_initialization(normalization_factor: float) -> Callable[[Tuple], ndarray]:

    return lambda shape: numpy.random.uniform(-normalization_factor, normalization_factor, shape)