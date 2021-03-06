import numpy
from theano import tensor as T
import models
from datasets import SharedDataset
from model_functions import compile_batch_training_function
from models import Classifier, Model
from training_step_evaluation import TrainingStepEvaluationStrategy

class StochasticGradientDescent:

    def __init__(self,
                 model: Model,
                 training_set: SharedDataset,
                 cost: T.TensorVariable,
                 learning_rate: float,
                 batch_size: int,
                 num_epochs: int,
                 evaluation_strategy: TrainingStepEvaluationStrategy):

        self.model = model
        self.num_training_batches = training_set.size // batch_size
        self.num_epochs = num_epochs

        self.train = compile_batch_training_function(
            model,
            cost,
            learning_rate,
            training_set,
            batch_size
        )

        self.evaluation_strategy = evaluation_strategy

    def run(self, save_path: str):

        epoch = 0
        iteration = 0

        current_loss = numpy.inf

        while epoch < self.num_epochs:

            epoch += 1

            for batch_index in range(self.num_training_batches):

                iteration += 1

                current_loss = self.train(batch_index)

                print('Iteration %i, loss %f%%' % (iteration, current_loss * 100))

            is_new_best, stopping_criterion_met = self.evaluation_strategy.apply(
                epoch,
                iteration,
                current_loss
            )

            if is_new_best:

                models.save(self.model, save_path)

            if stopping_criterion_met:

                print('Early stopping criterion met.')

                return