from abc import ABCMeta, abstractmethod
import numpy
from theano import tensor as T
import models
from models import Model
from datasets import SharedDataset
from model_functions import compile_testing_function


class TrainingStepEvaluationStrategy(metaclass=ABCMeta):

    def __init__(self,
                 model: Model):

        self.model = model

    best_loss = numpy.inf

    def is_new_best(self, loss: float) -> bool:

        return loss < self.best_loss

    def update_best_loss(self, loss: float):

        self.best_loss = loss

    def save_model(self, save_path: str):

        models.save(self.model, save_path)

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
              loss: float,
              save_path: str) -> bool:

        if self.is_new_best(loss):

            self.update_best_loss(loss)
            self.save_model(save_path)

        self.log_progress(epoch, iteration, loss)

        return self.stopping_criterion_met(iteration)


class NoEarlyStopping(TrainingStepEvaluationStrategy):

    def stopping_criterion_met(self, iteration: int):

        return False


class PatienceBasedEarlyStopping(TrainingStepEvaluationStrategy):

    def __init__(self,
                 model: Model,
                 validation_set: SharedDataset,
                 validation_cost: T.TensorVariable,
                 patience: int,
                 improvement_threshold: float,
                 patience_increase: float):

        super().__init__(model)

        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.patience_increase = patience_increase

        self.validate = compile_testing_function(model, validation_cost, validation_set)

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
              loss: float,
              save_path: str) -> bool:

        validation_loss = self.validate()

        if self.is_new_best(validation_loss):

            self.save_model(save_path)
            self.update_patience(validation_loss, iteration)
            self.update_best_loss(validation_loss)

        self.log_progress(epoch, iteration, validation_loss)

        return self.stopping_criterion_met(iteration)