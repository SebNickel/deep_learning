from typing import Callable, List
from theano import tensor as T
from models import Model, Classifier


def zero_one_losses(classifier: Classifier) -> T.TensorVariable:

    return T.neq(classifier.prediction, classifier.y)


def mean_zero_one_loss(classifier: Classifier) -> T.TensorVariable:

    return T.mean(zero_one_losses(classifier))


def individual_log_likelihoods(model: Model) -> T.TensorVariable:

    log_probability_matrix = T.log(model.output)

    return log_probability_matrix[T.arange(model.y.shape[0]), model.y]


def negative_log_likelihood(model: Model) -> T.TensorVariable:

    return -T.sum(individual_log_likelihoods(model))


def mean_negative_log_likelihood(model: Model) -> T.TensorVariable:

    return -T.mean(individual_log_likelihoods(model))


def composition(model: Model,
                cost_function: Callable[[Model], T.TensorVariable],
                regularization_weights: List[float],
                regularization_functions: List[Callable[[Model], T.TensorVariable]]) -> T.TensorVariable:

    cost = cost_function(model)

    regularization_terms = [function(model) for function in regularization_functions]

    weighted_regularization_terms = [weight * term for weight, term in zip(regularization_weights, regularization_terms)]

    return cost + T.sum(weighted_regularization_terms)


def compose(cost_function: Callable[[Model], T.TensorVariable],
            regularization_weights: List[float],
            regularization_functions: List[Callable[[Model], T.TensorVariable]]) -> Callable[[Model], T.TensorVariable]:

    return lambda model: composition(model, cost_function, regularization_weights, regularization_functions)