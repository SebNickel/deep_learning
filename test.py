import timeit
import numpy
from theano import tensor as T
import datasets
import initialization
import models
from cost import mean_negative_log_likelihood, mean_zero_one_loss, compose
from datasets import SharedDataset
from model_functions import compile_testing_function, compile_batch_testing_function
from models import Model, LinearModel, ActivationLayer, GeneralizedLinearModel, MultiLayerPerceptron, ConvolutionLayer, PoolingLayer, ConvolutionalNeuralNetwork
from regularization import l2_squared
from stochastic_gradient_descent import StochasticGradientDescent
from training_step_evaluation import TrainingStepEvaluationStrategy, PatienceBasedEarlyStopping


def train(model: Model,
          training_set: SharedDataset,
          cost: T.TensorVariable,
          learning_rate: float,
          batch_size: int,
          num_epochs: int,
          evaluation_strategy: TrainingStepEvaluationStrategy,
          save_path: str):

    print('Instantiating SGD class.')

    sgd = StochasticGradientDescent(
        model,
        training_set,
        cost,
        learning_rate,
        batch_size,
        num_epochs,
        evaluation_strategy
    )

    start_time = timeit.default_timer()

    print('Start training.')

    sgd.run(save_path)

    end_time = timeit.default_timer()

    duration = end_time - start_time

    print('Finished training after %f seconds.' % duration)


def test(model: Model,
         cost: T.TensorVariable,
         test_set: SharedDataset) -> float:

    test = compile_testing_function(model, cost, test_set)

    loss = test()

    return loss


def test_batchwise(
        model: Model,
        cost: T.TensorVariable,
        test_set: SharedDataset,
        batch_size: int) -> float:

    test_batch = compile_batch_testing_function(model, cost, test_set, batch_size)

    num_batches = test_set.size // batch_size

    losses = [test_batch(index) for index in range(num_batches)]

    mean_loss = numpy.mean(losses)

    return mean_loss


if __name__ == '__main__':

    print('Loading data.')

    training_set_path = 'mnist_train.pkl'
    validation_set_path = 'mnist_validate.pkl'
    test_set_path = 'mnist_test.pkl'

    save_path = 'mnist_model2.pkl'

    training_set = datasets.load(training_set_path)
    validation_set = datasets.load(validation_set_path)
    test_set = datasets.load(test_set_path)

    print('Initializing model.')

    random_number_generator = numpy.random.RandomState(23455)

    batch_size = 500

    num_input_channels_1 = 1
    input_height_1 = 28
    input_width_1 = 28

    num_filters_1 = 20
    filter_height_1 = 5
    filter_width_1 = 5

    fan_in_1 = num_input_channels_1 * input_height_1 * input_width_1
    fan_out_1 = num_filters_1 * filter_height_1 * filter_width_1

    normalization_factor_1 = initialization.normalization_factor_for_tanh(fan_in_1, fan_out_1)
    weight_initialization_strategy_1 = initialization.uniform_initialization(
        normalization_factor_1,
        random_number_generator
    )

    convolutional_layer_1 = ConvolutionLayer.build(
        input_height=input_height_1,
        input_width=input_width_1,
        filter_height=filter_height_1,
        filter_width=filter_width_1,
        num_input_channels=num_input_channels_1,
        num_filters=num_filters_1,
        batch_size=batch_size,
        weight_initialization=weight_initialization_strategy_1,
        bias_initialization=initialization.zero_initialization()
    )

    max_pooling_layer_1 = PoolingLayer(filter_height=2, filter_width=2)

    tanh_activation_layer_1 = ActivationLayer(activation_function=T.tanh)

    num_input_channels_2 = 20
    input_height_2 = 12
    input_width_2 = 12

    num_filters_2 = 50
    filter_height_2 = 5
    filter_width_2 = 5

    fan_in_2 = num_input_channels_2 * input_height_2 * input_width_2
    fan_out_2 = num_filters_2 * filter_height_2 * filter_width_2

    normalization_factor_2 = initialization.normalization_factor_for_tanh(fan_in_2, fan_out_2)
    weight_initialization_strategy_2 = initialization.uniform_initialization(
        normalization_factor_2,
        random_number_generator
    )

    convolutional_layer_2 = ConvolutionLayer.build(
        input_height=input_height_2,
        input_width=input_width_2,
        filter_height=filter_height_2,
        filter_width=filter_width_2,
        num_input_channels=num_input_channels_2,
        num_filters=num_filters_2,
        batch_size=batch_size,
        weight_initialization=weight_initialization_strategy_2,
        bias_initialization=initialization.zero_initialization()
    )

    max_pooling_layer_2 = PoolingLayer(filter_height=2, filter_width=2)

    tanh_activation_layer_2 = ActivationLayer(activation_function=T.tanh)

    cnn_layers = [
        convolutional_layer_1,
        max_pooling_layer_1,
        tanh_activation_layer_1,
        convolutional_layer_2,
        max_pooling_layer_2,
        tanh_activation_layer_2
    ]

    input_dim = 50 * 4 * 4
    num_hidden_units = 500
    output_dim = 10

    normalization_factor = initialization.normalization_factor_for_tanh(input_dim, num_hidden_units)

    linear_model_1 = LinearModel(
        input_dim=input_dim,
        output_dim=num_hidden_units,
        weight_initialization=initialization.uniform_initialization(normalization_factor),
        bias_initialization=initialization.zero_initialization()
    )

    tanh_activation_layer_3 = ActivationLayer(activation_function=T.tanh)

    linear_model_2 = LinearModel(
        input_dim=num_hidden_units,
        output_dim=output_dim,
        weight_initialization=initialization.zero_initialization(),
        bias_initialization=initialization.zero_initialization()
    )

    softmax_activation_layer = ActivationLayer(activation_function=T.nnet.softmax)

    generalized_linear_model_1 = GeneralizedLinearModel(linear_model_1, tanh_activation_layer_3)
    generalized_linear_model_2 = GeneralizedLinearModel(linear_model_2, softmax_activation_layer)

    multi_layer_perceptron = MultiLayerPerceptron(generalized_linear_model_1, generalized_linear_model_2)

    cnn = ConvolutionalNeuralNetwork(
        cnn_layers,
        multi_layer_perceptron
    )

    training_cost_function = compose(
        cost_function=mean_negative_log_likelihood,
        regularization_parameters=[0.0001],
        regularization_functions=[l2_squared]
    )

    training_cost = training_cost_function(cnn)

    print('Setting up early stopping strategy.')

    validation_cost = mean_zero_one_loss(cnn)

    evaluation_strategy = PatienceBasedEarlyStopping(
        cnn,
        validation_set,
        validation_cost,
        batch_size=batch_size,
        patience=10000,
        improvement_threshold=0.995,
        patience_increase=2
    )

    train(
        cnn,
        training_set,
        training_cost,
        learning_rate=0.1,
        batch_size=batch_size,
        num_epochs=200,
        evaluation_strategy=evaluation_strategy,
        save_path=save_path
    )

    print('Running test.')

    trained_model = models.load(save_path)

    test_cost = mean_zero_one_loss(trained_model)

    loss = test_batchwise(
        trained_model,
        test_cost,
        test_set,
        batch_size
    )

    print('Mean zero-one loss: %f%%' % (loss * 100))