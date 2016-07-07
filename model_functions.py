import theano
from theano import tensor as T
from theano.compile.function_module import Function
from models import GeneralizedLinearModel


def compile_training_function(model: GeneralizedLinearModel,
                              cost: T.TensorVariable,
                              learning_rate: float) -> Function:

    grad_W = T.grad(
        cost=cost,
        wrt=model.W
    )

    grad_b = T.grad(
        cost=cost,
        wrt=model.b
    )

    updates = [
        (model.W, model.W - learning_rate * grad_W),
        (model.b, model.b - learning_rate * grad_b)
    ]

    return theano.function(
        inputs=[model.x, model.y],
        outputs=cost,
        updates=updates,
        allow_input_downcast=True
    )


def compile_testing_function(model: GeneralizedLinearModel,
                             cost: T.TensorVariable) -> Function:

    return theano.function(
        inputs=[model.x, model.y],
        outputs=cost,
        allow_input_downcast=True
    )


def compile_prediction_function(model: GeneralizedLinearModel) -> Function:

    return theano.function(
        inputs=[model.x],
        outputs=model.prediction
    )
