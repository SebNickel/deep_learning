import pickle
from abc import ABCMeta, abstractproperty
from typing import Callable, Tuple, List
import theano
from numpy import ndarray
from theano import tensor as T
from theano.compile.function_module import Function
from theano.tensor.signal.pool import pool_2d
from utils import flatten


class Model(metaclass=ABCMeta):

    input = T.TensorVariable
    labels = T.TensorVariable

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

        return self.activation_function(self.input)


class LinearModel(Model):

    input = T.dmatrix('input')

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

        return T.dot(self.input, self.W) + self.b


class GeneralizedLinearModel(Classifier):

    input = T.dmatrix('input')
    labels = T.ivector('labels')

    def wire_up(self):

        self.linear_layer.input = self.input
        self.activation_layer.input = self.linear_layer.output

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

    input = T.dmatrix('input')
    labels = T.ivector('labels')

    def wire_up(self):

        self.layers[0].input = self.input
        self.layers[0].wire_up()

        for index in range(1, len(self.layers)):

            self.layers[index].input = self.layers[index - 1].output
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


class ConvolutionLayer(Model):

    input = T.tensor4('input')

    def __init__(self,
                 input_height: int,
                 input_width: int,
                 filter_height: int,
                 filter_width: int,
                 num_input_channels: int,
                 num_filters: int,
                 batch_size: int,
                 weight_initialization: Callable[[Tuple], ndarray],
                 bias_initialization: Callable[[Tuple], ndarray]):

        self.input_tensor_shape = (batch_size, num_input_channels, input_height, input_width)
        self.filter_tensor_shape = (num_filters, num_input_channels, filter_height, filter_width)

        self.W = theano.shared(
            value=weight_initialization(self.filter_tensor_shape),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=bias_initialization((num_filters,)),
            name='b',
            borrow=True
        )

    @property
    def weights(self):

        return [self.W]

    @property
    def params(self):

        return [self.W, self.b]

    @property
    def output(self):

        convolution = T.nnet.conv2d(
            input=self.input,
            filters=self.W,
            filter_shape=self.filter_tensor_shape,
            input_shape=self.input_tensor_shape
        )

        return convolution + self.b.dimshuffle('x', 0, 'x', 'x')


class MaxPoolingLayer(Model):

    input = T.tensor4('input')

    def __init__(self,
                 filter_height: int,
                 filter_width: int,
                 ignore_border: bool=True):

        self.filter_shape = (filter_height, filter_width)
        self.ignore_border = ignore_border

    @property
    def weights(self):

        return []

    @property
    def params(self):

        return []

    @property
    def output(self):

        return pool_2d(
            input=self.input,
            ds=self.filter_shape,
            ignore_border=self.ignore_border,
            mode='max'
        )


class MaxConvLayer(Model):

    input = T.tensor4('input')

    def wire_up(self):

        self.convolution_layer.input = self.input
        self.max_pooling_layer.input = self.convolution_layer.output
        self.activation_layer.input = self.max_pooling_layer.output

    def __init__(self,
                 convolution_layer: ConvolutionLayer,
                 max_pooling_layer: MaxPoolingLayer,
                 activation_layer: ActivationLayer):

        self.convolution_layer = convolution_layer
        self.max_pooling_layer = max_pooling_layer
        self.activation_layer = activation_layer

        self.wire_up()

    @property
    def weights(self):

        return self.convolution_layer.weights

    @property
    def params(self):

        return self.convolution_layer.params

    @property
    def output(self):

        return self.activation_layer.output


class LeNetModel(Classifier):

    input = T.dmatrix('input')
    labels = T.ivector('labels')

    def wire_up(self):

        input_tensor_shape = self.max_conv_layers[0].convolution_layer.input_tensor_shape

        reshaped_input = self.input.reshape(input_tensor_shape)

        self.max_conv_layers[0].input = reshaped_input
        self.max_conv_layers[0].wire_up()

        for index in range(1, len(self.max_conv_layers)):

            self.max_conv_layers[index].input = self.max_conv_layers[index - 1].output
            self.max_conv_layers[index].wire_up()

        flattened_output = self.max_conv_layers[-1].output.flatten(2)

        self.multi_layer_perceptron.input = flattened_output
        self.multi_layer_perceptron.wire_up()

    def __init__(self,
                 max_conv_layers: List[MaxConvLayer],
                 multi_layer_perceptron: MultiLayerPerceptron):

        self.max_conv_layers = max_conv_layers
        self.multi_layer_perceptron = multi_layer_perceptron

        self.wire_up()

    @property
    def weights(self):

        convolution_weights = flatten([max_conv_layer.weights for max_conv_layer in self.max_conv_layers])
        perceptron_weights = self.multi_layer_perceptron.weights

        return convolution_weights + perceptron_weights

    @property
    def params(self):

        convolution_params = flatten([max_conv_layer.params for max_conv_layer in self.max_conv_layers])
        perceptron_params = self.multi_layer_perceptron.params

        return convolution_params + perceptron_params

    @property
    def output(self):

        return self.multi_layer_perceptron.output


def load(file_path: str) -> Model:

    with open(file_path, 'rb') as file:

        return pickle.load(file)


def save(model: Model,
         target_path: str):

    with open(target_path, 'wb') as file:

        pickle._dump(model, file)