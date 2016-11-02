from abc import ABCMeta, abstractmethod
import numpy
from theano import tensor as T
from datasets import SharedDataset
from model_functions import compile_batch_testing_function
from models import Model, Classifier


class TrainingStepEvaluationStrategy(metaclass=ABCMeta):

    def __init__(self,
                 model: Model):

        self.model = model

    best_loss = numpy.inf

    def is_new_best(self, loss: float) -> bool:

        return loss < self.best_loss

    def update_best_loss(self, loss: float):

        self.best_loss = loss

    def log_progress(self,
                     epoch: int,
                     iteration: int,
                     loss: float):

        print(
            'Epoch %i, overall iteration %i, loss %f%%, current best %f%%' %
            (
                epoch,
                iteration,
                loss * 100,
                self.best_loss * 100
            )
        )

    @abstractmethod
    def stopping_criterion_met(self, iteration: int) -> bool:

        pass

    def apply(self,
              epoch: int,
              iteration: int,
              loss: float) -> (bool, bool):

        is_new_best = self.is_new_best(loss)
        stopping_criterion_met = self.stopping_criterion_met(iteration)

        if is_new_best:
            self.update_best_loss(loss)

        self.log_progress(epoch, iteration, loss)

        return is_new_best, stopping_criterion_met


class NoEarlyStopping(TrainingStepEvaluationStrategy):

    def stopping_criterion_met(self, iteration: int):

        return False


class PatienceBasedEarlyStopping(TrainingStepEvaluationStrategy):

    def __init__(self,
                 classifier: Classifier,
                 validation_set: SharedDataset,
                 validation_cost: T.TensorVariable,
                 batch_size: int,
                 patience: int,
                 improvement_threshold: float,
                 patience_increase: float):

        super().__init__(classifier)

        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.patience_increase = patience_increase

        self.validate = compile_batch_testing_function(classifier, validation_cost, validation_set, batch_size)
        self.num_validation_batches = validation_set.size // batch_size

    def update_patience(self,
                        validation_loss: float,
                        iteration: int):

        if validation_loss < self.best_loss * self.improvement_threshold:

            self.patience = max(self.patience, iteration * self.patience_increase)

    def log_progress(self,
                     epoch: int,
                     iteration: int,
                     validation_loss: float):

        print(
            'Epoch %i, overall iteration %i, validation loss %f%%, current best %f%%' %
            (
                epoch,
                iteration,
                validation_loss * 100,
                self.best_loss * 100
            )
        )

    def stopping_criterion_met(self, iteration: int) -> bool:

        return self.patience <= iteration

    def apply(self,
              epoch: int,
              iteration: int,
              loss: float) -> (bool, bool):

        validation_losses = [self.validate(index) for index in range(self.num_validation_batches)]

        mean_validation_loss = numpy.mean(validation_losses)

        is_new_best = self.is_new_best(mean_validation_loss)
        stopping_criterion_met = self.stopping_criterion_met(iteration)

        if self.is_new_best(mean_validation_loss):

            self.update_patience(mean_validation_loss, iteration)
            self.update_best_loss(mean_validation_loss)

        self.log_progress(epoch, iteration, mean_validation_loss)

        return is_new_best, stopping_criterion_met