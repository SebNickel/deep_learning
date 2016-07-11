from typing import Callable, Tuple, List
from abc import ABCMeta, abstractproperty
import pickle
import theano
from theano import tensor as T
from theano.compile.function_module import Function
from numpy import ndarray
from utils import flatten


class Model(metaclass=ABCMeta):

    x = T.dmatrix('x')
    y = T.ivector('y')

    @abstractproperty
    def weights(self) -> List[T.TensorVariable]:

        pass

    @abstractproperty
    def params(self) -> List[T.TensorVariable]:

        pass

    @abstractproperty
    def linear_projection(self) -> T.TensorVariable:

        pass

    @abstractproperty
    def response(self) -> T.TensorVariable:

        pass

    @abstractproperty
    def prediction(self) -> T.TensorVariable:

        pass


class GeneralizedLinearModel(Model):

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
    def weights(self) -> List[T.TensorVariable]:

        return [self.W]

    @property
    def params(self) -> List[T.TensorVariable]:

        return [self.W, self.b]

    @property
    def linear_projection(self) -> T.TensorVariable:

        return T.dot(self.x, self.W) + self.b

    @property
    def response(self) -> T.TensorVariable:

        return self.link_function(self.linear_projection)

    @property
    def prediction(self) -> T.TensorVariable:

        return T.argmax(self.response, axis=1)


class MultiLayerPerceptron(Model):

    def __init__(self, *layers: List[GeneralizedLinearModel]):

        self.layers = layers

        self.input_layer = self.layers[0]
        self.hidden_layers = self.layers[1:-1]
        self.output_layer = self.layers[-1]

        self.x = self.input_layer.x
        self.y = self.output_layer.y

        for index in range(1, len(self.layers)):

            self.layers[index].x = self.layers[index - 1].response

    @property
    def weights(self) -> List[T.TensorVariable]:

        return [layer.W for layer in self.layers]

    @property
    def params(self) -> List[T.TensorVariable]:

        layer_params = [layer.params for layer in self.layers]

        return flatten(layer_params)

    @property
    def linear_projection(self) -> T.TensorVariable:

        return self.output_layer.linear_projection

    @property
    def response(self) -> T.TensorVariable:

        return self.output_layer.response

    @property
    def prediction(self) -> T.TensorVariable:

        return self.output_layer.prediction


def load(file_path: str) -> Model:

    with open(file_path, 'rb') as file:

        return pickle.load(file)


def save(model: Model,
         target_path: str):

    with open(target_path, 'wb') as file:

        pickle._dump(model, file)