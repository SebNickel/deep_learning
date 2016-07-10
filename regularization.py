from theano import tensor as T
from models import GeneralizedLinearModel


def l1(model: GeneralizedLinearModel) -> T.TensorVariable:

    return T.sum(abs(model.W)) + T.sum(abs(model.b))


def l2_squared(model: GeneralizedLinearModel) -> T.TensorVariable:

    return T.sum(model.W ** 2 + model.b ** 2)


def l2(model: GeneralizedLinearModel) -> T.TensorVariable:

    return T.sqrt(l2_squared(model))