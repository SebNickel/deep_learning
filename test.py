import timeit
import numpy
from theano import tensor as T
import datasets
import initialization
import models
from cost import mean_negative_log_likelihood, mean_zero_one_loss, compose
from datasets import SharedDataset
from model_functions import compile_testing_function, compile_batch_testing_function
from models import Model, Classifier, LinearModel, ActivationLayer, GeneralizedLinearModel, MultiLayerPerceptron, ConvolutionLayer, MaxPoolingLayer, MaxConvLayer, LeNetModel
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

    sgd = StochasticGradientDescent(model,
                                    training_set,
                                    cost,
                                    learning_rate,
                                    batch_size,
                                    num_epochs,
                                    evaluation_strategy)

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

    save_path = 'mnist_model.pkl'

    training_set = datasets.load(training_set_path)
    validation_set = datasets.load(validation_set_path)
    test_set = datasets.load(test_set_path)

    print('Initializing model.')

    random_number_generator = numpy.random.RandomState(23455)

    batch_size = 500

    first_num_input_channels = 1
    first_input_height = 28
    first_input_width = 28

    first_num_filters = 20
    first_filter_height = 5
    first_filter_width = 5

    first_fan_in = first_num_input_channels * first_input_height * first_input_width
    first_fan_out = first_num_filters * first_filter_height * first_filter_width

    first_normalization_factor = initialization.normalization_factor_for_tanh(first_fan_in, first_fan_out)
    first_weight_initialization_strategy = initialization.uniform_initialization(
        first_normalization_factor,
        random_number_generator
    )

    first_convolutional_layer = ConvolutionLayer(
        input_height=first_input_height,
        input_width=first_input_width,
        filter_height=first_filter_height,
        filter_width=first_filter_width,
        num_input_channels=first_num_input_channels,
        num_filters=first_num_filters,
        batch_size=batch_size,
        weight_initialization=first_weight_initialization_strategy,
        bias_initialization=initialization.zero_initialization()
    )

    second_num_input_channels = 20
    second_input_height = 12
    second_input_width = 12

    second_num_filters = 50
    second_filter_height = 5
    second_filter_width = 5

    second_fan_in = second_num_input_channels * second_input_height * second_input_width
    second_fan_out = second_num_filters * second_filter_height * second_filter_width

    second_normalization_factor = initialization.normalization_factor_for_tanh(second_fan_in, second_fan_out)
    second_weight_initialization_strategy = initialization.uniform_initialization(
        second_normalization_factor,
        random_number_generator
    )

    second_convolutional_layer = ConvolutionLayer(
        input_height=second_input_height,
        input_width=second_input_width,
        filter_height=second_filter_height,
        filter_width=second_filter_width,
        num_input_channels=second_num_input_channels,
        num_filters=second_num_filters,
        batch_size=batch_size,
        weight_initialization=second_weight_initialization_strategy,
        bias_initialization=initialization.zero_initialization()
    )

    max_pooling_layer = MaxPoolingLayer(filter_height=2, filter_width=2)

    tanh_activation_layer = ActivationLayer(activation_function=T.tanh)

    first_maxconv_layer = MaxConvLayer(first_convolutional_layer, max_pooling_layer, tanh_activation_layer)
    second_maxconv_layer = MaxConvLayer(second_convolutional_layer, max_pooling_layer, tanh_activation_layer)

    input_dim = 50 * 4 * 4
    num_hidden_units = 500
    output_dim = 10

    normalization_factor = initialization.normalization_factor_for_tanh(input_dim, num_hidden_units)

    first_linear_model = LinearModel(
        input_dim=input_dim,
        output_dim=num_hidden_units,
        weight_initialization=initialization.uniform_initialization(normalization_factor),
        bias_initialization=initialization.zero_initialization()
    )

    second_linear_model = LinearModel(
        input_dim=num_hidden_units,
        output_dim=output_dim,
        weight_initialization=initialization.zero_initialization(),
        bias_initialization=initialization.zero_initialization()
    )

    softmax_activation_layer = ActivationLayer(activation_function=T.nnet.softmax)

    first_generalized_linear_model = GeneralizedLinearModel(first_linear_model, tanh_activation_layer)
    second_generalized_linear_model = GeneralizedLinearModel(second_linear_model, softmax_activation_layer)

    multi_layer_perceptron = MultiLayerPerceptron(first_generalized_linear_model, second_generalized_linear_model)

    lenet_model = LeNetModel(
        max_conv_layers=[first_maxconv_layer, second_maxconv_layer],
        multi_layer_perceptron=multi_layer_perceptron
    )

    training_cost_function = compose(
        cost_function=mean_negative_log_likelihood,
        regularization_parameters=[0.0001],
        regularization_functions=[l2_squared]
    )

    training_cost = training_cost_function(lenet_model)

    print('Setting up early stopping strategy.')

    validation_cost = mean_zero_one_loss(lenet_model)

    evaluation_strategy = PatienceBasedEarlyStopping(
        lenet_model,
        validation_set,
        validation_cost,
        batch_size=batch_size,
        patience=10000,
        improvement_threshold=0.995,
        patience_increase=2
    )

    train(
        lenet_model,
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