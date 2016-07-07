import numpy
from theano import tensor as T
from datasets import Dataset
from iteration import batch_iterator
from model_functions import compile_training_function, compile_testing_function
from models import GeneralizedLinearModel, save


class TrainingStepEvaluation:

    def __init__(self,
                 patience: int,
                 improvement_threshold: float,
                 patience_increase: float):

        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.patience_increase = patience_increase

    best_validation_loss = numpy.inf

    def is_new_best(self, validation_loss: float) -> bool:

        return validation_loss < self.best_validation_loss

    def update_patience(self,
                        validation_loss: float,
                        iteration: int):

        if validation_loss < self.best_validation_loss * self.improvement_threshold:

            self.patience = max(self.patience, iteration * self.patience_increase)

    def update_best_validation_loss(self, validation_loss: float):

        self.best_validation_loss = validation_loss

    def stopping_criterion_met(self, iteration: int) -> bool:

        return self.patience <= iteration


class SGD:

    def __init__(self,
                 model: GeneralizedLinearModel,
                 training_set: Dataset,
                 validation_set: Dataset,
                 training_cost: T.TensorVariable,
                 validation_cost: T.TensorVariable,
                 learning_rate: float,
                 batch_size: int,
                 num_epochs: int,
                 patience: int,
                 improvement_threshold: float,
                 patience_increase: float,
                 save_path: str):

        self.model = model

        self.training_set = training_set
        self.validation_set = validation_set

        self.batch_size = batch_size
        self.num_training_batches = training_set.vectors.shape[0] // batch_size

        self.num_epochs = num_epochs

        self.save_path = save_path

        self.train = compile_training_function(model, training_cost, learning_rate)
        self.validate = compile_testing_function(model, validation_cost)

        self.validation_frequency = min(self.num_training_batches, patience // 2)

        self.training_step_evaluation = TrainingStepEvaluation(patience, improvement_threshold, patience_increase)

    @staticmethod
    def log_progress(epoch: int,
                     iteration: int,
                     validation_loss: float):

        print(
            'Epoch %i, overall iteration %i, validation loss %f%%' %
            (
                epoch,
                iteration,
                validation_loss * 100
            )
        )

    def run(self):

        epoch = 0
        iteration = 0

        while epoch < self.num_epochs:

            epoch += 1

            batches = batch_iterator(self.training_set, self.batch_size)

            for x_train, y_train in batches:

                iteration += 1

                self.train(x_train, y_train)

                if iteration % self.validation_frequency == 0:

                    x_validate = self.validation_set.vectors
                    y_validate = self.validation_set.labels

                    validation_loss = self.validate(x_validate, y_validate)

                    self.log_progress(epoch, iteration, validation_loss)

                    if self.training_step_evaluation.is_new_best(validation_loss):

                        self.training_step_evaluation.update_patience(validation_loss, iteration)
                        self.training_step_evaluation.update_best_validation_loss(validation_loss)

                        save(self.model, self.save_path)

                if self.training_step_evaluation.stopping_criterion_met(iteration):

                    print('Stopping criterion met.')

                    return

        print('Completed %i epochs without meeting stopping criterion.' % self.num_epochs)
