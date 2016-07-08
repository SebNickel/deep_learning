import theano
from theano import tensor as T
from theano.compile.function_module import Function
from models import GeneralizedLinearModel
from datasets import SharedDataset


def compile_batch_training_function(model: GeneralizedLinearModel,
                                    cost: T.TensorVariable,
                                    learning_rate: float,
                                    dataset: SharedDataset,
                                    batch_size: int) -> Function:

    batch_index = T.lscalar('batch_index')

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
        inputs=[batch_index],
        outputs=cost,
        updates=updates,
        givens={
            model.x: dataset.x[batch_index * batch_size: (batch_index + 1) * batch_size],
            model.y: dataset.y[batch_index * batch_size: (batch_index + 1) * batch_size]
        },
        allow_input_downcast=True
    )


def compile_testing_function(model: GeneralizedLinearModel,
                             cost: T.TensorVariable,
                             dataset: SharedDataset) -> Function:

    return theano.function(
        inputs=[],
        outputs=cost,
        givens={
            model.x: dataset.x,
            model.y: dataset.y
        },
        allow_input_downcast=True
    )


def compile_response_function(model: GeneralizedLinearModel) -> Function:

    return theano.function(
        inputs=[model.x],
        outputs=model.response
    )


def compile_prediction_function(model: GeneralizedLinearModel) -> Function:

    return theano.function(
        inputs=[model.x],
        outputs=model.prediction
    )