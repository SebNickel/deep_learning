from typing import Callable, Tuple, List
import pickle
import theano
from theano import tensor as T
from theano.compile.function_module import Function
from numpy import ndarray


class GeneralizedLinearModel:

    x = T.dmatrix('x')
    y = T.ivector('y')

    def __init__(self,
                 input_dim: int,
                 linear_output_dim: int,
                 link_function: Function,
                 weight_initialization: Callable[[Tuple], ndarray],
                 bias_initialization: Callable[[Tuple], ndarray]):

        self.input_dim = input_dim
        self.linear_output_dim = linear_output_dim
        self.link_function = link_function

        W_shape = (input_dim, linear_output_dim)
        b_shape = (linear_output_dim,)

        self.W = theano.shared(
            value=weight_initialization(W_shape),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=bias_initialization(b_shape),
            name='b',
            borrow=True
        )

    @property
    def linear_projection(self):

        return T.dot(self.x, self.W) + self.b

    @property
    def response(self):

        return self.link_function(self.linear_projection)

    @property
    def prediction(self):

        return T.argmax(self.response, axis=1)


def load(file_path: str) -> GeneralizedLinearModel:

    with open(file_path, 'rb') as file:

        return pickle.load(file)


def save(model: GeneralizedLinearModel,
         target_path: str):

    with open(target_path, 'wb') as file:

        pickle._dump(model, file)