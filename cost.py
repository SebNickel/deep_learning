from theano import tensor as T
from models import GeneralizedLinearModel


def log_likelihoods(model: GeneralizedLinearModel) -> T.TensorVariable:

    log_probability_matrix = T.log(model.response)

    log_likelihoods = log_probability_matrix[T.arange(model.y.shape[0]), model.y]

    return log_likelihoods


def negative_log_likelihood(model: GeneralizedLinearModel) -> T.TensorVariable:

    individual_log_likelihoods = log_likelihoods(model)

    return -T.sum(individual_log_likelihoods)


def mean_negative_log_likelihood(model: GeneralizedLinearModel) -> T.TensorVariable:

    individual_log_likelihoods = log_likelihoods(model)

    return -T.mean(individual_log_likelihoods)


def mean_error(model: GeneralizedLinearModel) -> T.TensorVariable:

    return T.mean(T.neq(model.prediction, model.y))