from typing import Callable, List
from theano import tensor as T
from models import GeneralizedLinearModel


def zero_one_losses(model: GeneralizedLinearModel) -> T.TensorVariable:

    return T.neq(model.prediction, model.y)


def mean_zero_one_loss(model: GeneralizedLinearModel) -> T.TensorVariable:

    return T.mean(zero_one_losses(model))


def individual_log_likelihoods(model: GeneralizedLinearModel) -> T.TensorVariable:

    log_probability_matrix = T.log(model.response)

    return log_probability_matrix[T.arange(model.y.shape[0]), model.y]


def negative_log_likelihood(model: GeneralizedLinearModel) -> T.TensorVariable:

    return -T.sum(individual_log_likelihoods(model))


def mean_negative_log_likelihood(model: GeneralizedLinearModel) -> T.TensorVariable:

    return -T.mean(individual_log_likelihoods(model))


def composition(model: GeneralizedLinearModel,
                cost_function: Callable[[GeneralizedLinearModel], T.TensorVariable],
                regularization_weights: List[float],
                regularization_functions: List[Callable[[GeneralizedLinearModel], T.TensorVariable]]) -> T.TensorVariable:

    cost = cost_function(model)

    regularization_terms = [function(model) for function in regularization_functions]

    weighted_regularization_terms = [weight * term for weight, term in zip(regularization_weights, regularization_terms)]

    return cost + T.sum(weighted_regularization_terms)


def compose(cost_function: Callable[[GeneralizedLinearModel], T.TensorVariable],
            regularization_weights: List[float],
            regularization_functions: List[Callable[[GeneralizedLinearModel], T.TensorVariable]]) -> Callable[[GeneralizedLinearModel], T.TensorVariable]:

    return lambda model: composition(model, cost_function, regularization_weights, regularization_functions)