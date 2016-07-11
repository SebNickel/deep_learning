import theano
from theano import tensor as T
from theano.compile.function_module import Function
from models import Model
from datasets import SharedDataset


def compile_batch_training_function(model: Model,
                                    cost: T.TensorVariable,
                                    learning_rate: float,
                                    dataset: SharedDataset,
                                    batch_size: int) -> Function:

    batch_index = T.lscalar('batch_index')

    gradients = [
        T.grad(cost=cost, wrt=param)
        for param in model.params
    ]

    updates = [
        (param, param - learning_rate * gradient)
        for param, gradient in zip(model.params, gradients)
    ]

    return theano.function(
        inputs=[batch_index],
        outputs=cost,
        updates=updates,
        givens={
            model.x: dataset.x[batch_index * batch_size: (batch_index + 1) * batch_size],
            model.y: dataset.y[batch_index * batch_size: (batch_index + 1) * batch_size]
        }
    )


def compile_testing_function(model: Model,
                             cost: T.TensorVariable,
                             dataset: SharedDataset) -> Function:

    return theano.function(
        inputs=[],
        outputs=cost,
        givens={
            model.x: dataset.x,
            model.y: dataset.y
        }
    )


def compile_response_function(model: Model) -> Function:

    return theano.function(
        inputs=[model.x],
        outputs=model.response
    )


def compile_prediction_function(model: Model) -> Function:

    return theano.function(
        inputs=[model.x],
        outputs=model.prediction
    )