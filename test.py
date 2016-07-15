import timeit
from theano import tensor as T
import datasets
import initialization
import models
from cost import mean_negative_log_likelihood, mean_zero_one_loss, compose
from datasets import SharedDataset
from model_functions import compile_testing_function
from models import Model, Classifier, LinearModel, ActivationLayer, GeneralizedLinearModel, MultiLayerPerceptron
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


if __name__ == '__main__':

    input_dim = 28 * 28
    num_hidden_units = 500

    normalization_factor = initialization.normalization_factor_for_tanh(input_dim, num_hidden_units)

    print('Initializing model.')

    hidden_linear_layer = LinearModel(
        input_dim=28 * 28,
        output_dim=num_hidden_units,
        weight_initialization=initialization.uniform_initialization(normalization_factor),
        bias_initialization=initialization.zero_initialization()
    )

    hidden_activation_layer = ActivationLayer(T.tanh)

    hidden_layer = GeneralizedLinearModel(
        hidden_linear_layer,
        hidden_activation_layer
    )

    visible_linear_layer = LinearModel(
        input_dim=num_hidden_units,
        output_dim=10,
        weight_initialization=initialization.zero_initialization(),
        bias_initialization=initialization.zero_initialization()
    )

    visible_activation_layer = ActivationLayer(T.nnet.softmax)

    logistic_regression_layer = GeneralizedLinearModel(
        visible_linear_layer,
        visible_activation_layer
    )

    multi_layer_perceptron = MultiLayerPerceptron(hidden_layer, logistic_regression_layer)

    training_set_path = 'mnist_train.pkl'
    validation_set_path = 'mnist_validate.pkl'
    test_set_path = 'mnist_test.pkl'

    save_path = 'mnist_model.pkl'

    training_set = datasets.load(training_set_path)
    validation_set = datasets.load(validation_set_path)
    test_set = datasets.load(test_set_path)

    training_cost_function = compose(
        cost_function=mean_negative_log_likelihood,
        regularization_weights=[0.0001],
        regularization_functions=[l2_squared]
    )

    training_cost = training_cost_function(multi_layer_perceptron)

    validation_cost_function = mean_zero_one_loss

    validation_cost = validation_cost_function(multi_layer_perceptron)

    print('Setting up early stopping strategy.')

    evaluation_strategy = PatienceBasedEarlyStopping(
        multi_layer_perceptron,
        validation_set,
        validation_cost,
        patience=10000,
        improvement_threshold=0.995,
        patience_increase=2
    )

    train(multi_layer_perceptron,
          training_set,
          training_cost,
          learning_rate=0.01,
          batch_size=20,
          num_epochs=1000,
          evaluation_strategy=evaluation_strategy,
          save_path=save_path
          )

    print('Running test.')

    trained_model = models.load(save_path)

    test_cost = mean_zero_one_loss(trained_model)

    loss = test(trained_model,
                test_cost,
                test_set)

    print('Mean zero-one loss: %f%%' % (loss * 100))