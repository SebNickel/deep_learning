import timeit
from typing import Callable, Tuple
import numpy
from numpy import ndarray
from theano import tensor as T
from theano.compile.function_module import Function
import datasets
from cost import mean_negative_log_likelihood, mean_zero_one_loss, compose
from datasets import SharedDataset
from training_step_evaluation import TrainingStepEvaluationStrategy, PatienceBasedEarlyStopping, NoEarlyStopping
from model_functions import compile_testing_function
import models
from models import GeneralizedLinearModel
from sgd import SGD
from regularization import l2


def zero_initialize_model(vector_dim: int,
                          linear_output_dim: int,
                          link_function: Function):

    model = GeneralizedLinearModel(vector_dim, linear_output_dim, link_function)

    model.zero_initialize_weights()
    model.zero_initialize_bias()

    return model


def randomly_initialize_model(vector_dim: int,
                              linear_output_dim: int,
                              link_function: Function,
                              distribution: Callable[[Tuple], ndarray]):

    model = GeneralizedLinearModel(vector_dim, linear_output_dim, link_function)

    model.randomly_initialize_weights(distribution)
    model.randomly_initialize_bias(distribution)

    return model


def train(model: GeneralizedLinearModel,
          training_set: SharedDataset,
          cost: T.TensorVariable,
          learning_rate: float,
          batch_size: int,
          num_epochs: int,
          evaluation_strategy: TrainingStepEvaluationStrategy,
          save_path: str):

    print('Instantiating SGD class.')

    sgd = SGD(model,
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


def test(model: GeneralizedLinearModel,
         test_set: SharedDataset,
         cost: T.TensorVariable) -> float:

    test = compile_testing_function(model, cost, test_set)

    loss = test()

    return loss


if __name__ == '__main__':

    uniform_distribution = lambda shape: numpy.random.uniform(-0.5, 0.5, shape)

    print('Randomly initializing model.')

    logistic_regression_model = randomly_initialize_model(
        vector_dim=28 * 28,
        linear_output_dim=10,
        link_function=T.nnet.softmax,
        distribution=uniform_distribution
    )

    training_set_path = 'mnist_train.pkl'
    validation_set_path = 'mnist_validate.pkl'
    test_set_path = 'mnist_test.pkl'

    save_path = 'mnist_model.pkl'

    training_set = datasets.load(training_set_path)
    validation_set = datasets.load(validation_set_path)
    test_set = datasets.load(test_set_path)

    training_cost_function = compose(
        cost_function=mean_negative_log_likelihood,
        regularization_weights=[0.01],
        regularization_functions=[l2]
    )

    training_cost = training_cost_function(logistic_regression_model)

    validation_cost_function = mean_zero_one_loss

    validation_cost = validation_cost_function(logistic_regression_model)

    print('Setting up early stopping strategy.')

    evaluation_strategy = PatienceBasedEarlyStopping(
        logistic_regression_model,
        validation_set,
        validation_cost,
        patience=10000,
        improvement_threshold=0.995,
        patience_increase=2
    )

    train(logistic_regression_model,
          training_set,
          training_cost,
          learning_rate=0.13,
          batch_size=600,
          num_epochs=150,
          evaluation_strategy=evaluation_strategy,
          save_path=save_path
    )

    print('Running test.')

    trained_model = models.load(save_path)

    test_cost = mean_zero_one_loss(trained_model)

    loss = test(trained_model,
                test_set,
                test_cost)

    print('Mean zero-one loss: %f %%' % (loss * 100))