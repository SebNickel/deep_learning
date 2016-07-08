from typing import Callable
from theano import tensor as T
from theano.compile.function_module import Function
import timeit
from models import GeneralizedLinearModel
from sgd import SGD
import datasets
from datasets import SharedDataset
from model_functions import compile_testing_function
from cost import mean_negative_log_likelihood, mean_error


def train(training_set: SharedDataset,
          validation_set: SharedDataset,
          vector_dim: int,
          num_classes: int,
          link_function: Function,
          training_cost_function: Callable[[GeneralizedLinearModel], T.TensorVariable],
          validation_cost_function: Callable[[GeneralizedLinearModel], T.TensorVariable],
          learning_rate: float,
          batch_size: int,
          num_epochs: int,
          patience: int,
          improvement_threshold: float,
          patience_increase: float,
          save_path: str) -> GeneralizedLinearModel:

    print('Initialising model.')

    model = GeneralizedLinearModel(vector_dim, num_classes, link_function)

    model.zero_initialize_weights()
    model.zero_initialize_bias()

    training_cost = training_cost_function(model)
    validation_cost = validation_cost_function(model)

    print('Instantiating SGD class.')

    sgd = SGD(model,
              training_set,
              validation_set,
              training_cost,
              validation_cost,
              learning_rate,
              batch_size,
              num_epochs,
              patience,
              improvement_threshold,
              patience_increase,
              save_path)

    start_time = timeit.default_timer()

    print('Start training.')

    sgd.run()

    end_time = timeit.default_timer()

    duration = end_time - start_time

    print('Finished training after %f seconds.' % duration)

    return sgd.model


def test(model: GeneralizedLinearModel,
         test_set: SharedDataset,
         cost: T.TensorVariable) -> float:

    test = compile_testing_function(model, cost, test_set)

    loss = test()

    return loss


if __name__ == '__main__':

    training_set_path = 'mnist_train.pkl'
    validation_set_path = 'mnist_validate.pkl'
    test_set_path = 'mnist_test.pkl'

    training_set = datasets.load(training_set_path)
    validation_set = datasets.load(validation_set_path)
    test_set = datasets.load(test_set_path)

    logistic_regression_model = train(
        training_set,
        validation_set,
        vector_dim=28 * 28,
        num_classes=10,
        link_function=T.nnet.softmax,
        training_cost_function=mean_negative_log_likelihood,
        validation_cost_function=mean_error,
        learning_rate=0.13,
        batch_size=600,
        num_epochs=1000,
        patience=5000,
        improvement_threshold=0.995,
        patience_increase=2,
        save_path='mnist_model.pkl'
    )

    print('Running test.')

    test_cost = mean_error(logistic_regression_model)

    loss = test(logistic_regression_model,
                test_set,
                test_cost)

    print('Loss: %f %%' % (loss * 100))