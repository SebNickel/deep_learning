import theano
from theano import tensor as T
from theano.compile.function_module import Function
from models import Model, Classifier
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
            model.input: dataset.x[batch_index * batch_size: (batch_index + 1) * batch_size],
            model.labels: dataset.y[batch_index * batch_size: (batch_index + 1) * batch_size]
        }
    )


def compile_testing_function(model: Model,
                             cost: T.TensorVariable,
                             dataset: SharedDataset) -> Function:

    return theano.function(
        inputs=[],
        outputs=cost,
        givens={
            model.input: dataset.x,
            model.labels: dataset.y
        }
    )


def compile_batch_testing_function(model: Model,
                                   cost: T.TensorVariable,
                                   dataset: SharedDataset,
                                   batch_size: int) -> Function:

    batch_index = T.lscalar('batch_index')

    return theano.function(
        inputs=[batch_index],
        outputs=cost,
        givens={
            model.input: dataset.x[batch_index * batch_size: (batch_index + 1) * batch_size],
            model.labels: dataset.y[batch_index * batch_size: (batch_index + 1) * batch_size]
        }
    )


def compile_response_function(model: Model) -> Function:

    return theano.function(
        inputs=[model.input],
        outputs=model.output
    )


def compile_prediction_function(classifier: Classifier) -> Function:

    return theano.function(
        inputs=[classifier.input],
        outputs=classifier.prediction
    )