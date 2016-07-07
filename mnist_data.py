from typing import Tuple
import pickle
import gzip
from datasets import Dataset


def load(file_path: str='mnist.pkl.gz') -> Tuple[Dataset, Dataset, Dataset]:

    with gzip.open(file_path, 'rb') as file:

        try:

            train, validation, test = pickle.load(file, encoding='latin1')

        except:

            train, validation, test = pickle.load(file)

    training_vectors, training_labels = train
    validation_vectors, validation_labels = validation
    test_vectors, test_labels = test

    training_set = Dataset(training_vectors, training_labels)
    validation_set = Dataset(validation_vectors, validation_labels)
    test_set = Dataset(test_vectors, test_labels)

    return training_set, validation_set, test_set