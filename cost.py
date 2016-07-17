from typing import Callable, List
from theano import tensor as T
from models import Model, Classifier


def zero_one_losses(classifier: Classifier) -> T.TensorVariable:

    return T.neq(classifier.prediction, classifier.labels)


def mean_zero_one_loss(classifier: Classifier) -> T.TensorVariable:

    return T.mean(zero_one_losses(classifier))


def individual_log_likelihoods(classifier: Classifier) -> T.TensorVariable:

    log_probability_matrix = T.log(classifier.output)

    return log_probability_matrix[T.arange(classifier.labels.shape[0]), classifier.labels]


def negative_log_likelihood(classifier: Classifier) -> T.TensorVariable:

    return -T.sum(individual_log_likelihoods(classifier))


def mean_negative_log_likelihood(classifier: Classifier) -> T.TensorVariable:

    return -T.mean(individual_log_likelihoods(classifier))


def composition(classifier: Classifier,
                cost_function: Callable[[Classifier], T.TensorVariable],
                regularization_parameters: List[float],
                regularization_functions: List[Callable[[Model], T.TensorVariable]]) -> T.TensorVariable:

    cost = cost_function(classifier)

    unweighted_regularization_terms = [function(classifier) for function in regularization_functions]

    regularization_terms = [weight * term for weight, term in zip(regularization_parameters, unweighted_regularization_terms)]

    return cost + T.sum(regularization_terms)


def compose(cost_function: Callable[[Classifier], T.TensorVariable],
            regularization_parameters: List[float],
            regularization_functions: List[Callable[[Model], T.TensorVariable]]) -> Callable[[Classifier], T.TensorVariable]:

    return lambda model: composition(model, cost_function, regularization_parameters, regularization_functions)