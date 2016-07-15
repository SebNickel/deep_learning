import pickle
from abc import ABCMeta, abstractproperty
from typing import Callable, Tuple, List
import theano
from numpy import ndarray
from theano import tensor as T
from theano.compile.function_module import Function
from utils import flatten


class Model(metaclass=ABCMeta):

    x = T.TensorVariable
    y = T.TensorVariable

    @abstractproperty
    def weights(self) -> List[T.TensorVariable]:

        pass

    @abstractproperty
    def params(self) -> List[T.TensorVariable]:

        pass

    @abstractproperty
    def output(self) -> T.TensorVariable:

        pass


class Classifier(Model, metaclass=ABCMeta):

    @property
    def prediction(self) -> T.TensorVariable:

        return T.argmax(self.output, axis=1)


class ActivationLayer(Model):

    def __init__(self,
                 activation_function: Function):

        self.activation_function = activation_function

    @property
    def weights(self) -> List[T.TensorVariable]:

        return []

    @property
    def params(self) -> List[T.TensorVariable]:

        return []

    @property
    def output(self) -> T.TensorVariable:

        return self.activation_function(self.x)


class LinearModel(Model):

    x = T.dmatrix('x')
    y = T.ivector('y')

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 weight_initialization: Callable[[Tuple], ndarray],
                 bias_initialization: Callable[[Tuple], ndarray]):

        self.input_dim = input_dim
        self.output_dim = output_dim

        W_shape = (input_dim, output_dim)
        b_shape = (output_dim,)

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
    def output(self) -> T.TensorVariable:

        return T.dot(self.x, self.W) + self.b


class GeneralizedLinearModel(Classifier):

    x = T.dmatrix('x')
    y = T.ivector('y')

    def wire_up(self):

        self.linear_layer.x = self.x
        self.activation_layer.x = self.linear_layer.output

    def __init__(self,
                 linear_layer: LinearModel,
                 activation_layer: ActivationLayer):

        self.linear_layer = linear_layer
        self.activation_layer = activation_layer

        self.wire_up()

    @property
    def weights(self) -> List[T.TensorVariable]:

        return self.linear_layer.weights

    @property
    def params(self) -> List[T.TensorVariable]:

        return self.linear_layer.params

    @property
    def output(self) -> T.TensorVariable:

        return self.activation_layer.output


class MultiLayerPerceptron(Classifier):

    x = T.dmatrix('x')
    y = T.ivector('y')

    def wire_up(self):

        self.layers[0].x = self.x
        self.layers[0].wire_up()

        for index in range(1, len(self.layers)):

            self.layers[index].x = self.layers[index - 1].output
            self.layers[index].wire_up()

    def __init__(self, *layers: List[GeneralizedLinearModel]):

        self.layers = layers

        self.wire_up()

    @property
    def weights(self) -> List[T.TensorVariable]:

        layer_weights = [layer.weights for layer in self.layers]

        return flatten(layer_weights)

    @property
    def params(self) -> List[T.TensorVariable]:

        layer_params = [layer.params for layer in self.layers]

        return flatten(layer_params)

    @property
    def output(self) -> T.TensorVariable:

        return self.layers[-1].activation_layer.output


def load(file_path: str) -> Model:

    with open(file_path, 'rb') as file:

        return pickle.load(file)


def save(model: Model,
         target_path: str):

    with open(target_path, 'wb') as file:

        pickle._dump(model, file)