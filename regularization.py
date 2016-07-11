from theano import tensor as T
from models import Model


def l1(model: Model) -> T.TensorVariable:

    layer_wise_l1_norms = [abs(W).sum() for W in model.weights]

    return T.sum(layer_wise_l1_norms)


def l2_squared(model: Model) -> T.TensorVariable:

    layer_wise_l2_squared_norms = [(W ** 2).sum() for W in model.weights]

    return T.sum(layer_wise_l2_squared_norms)


def l2(model: Model) -> T.TensorVariable:

    return T.sqrt(l2_squared(model))