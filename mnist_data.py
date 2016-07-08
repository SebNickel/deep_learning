from typing import Tuple
import pickle
import gzip
from datasets import SharedDataset


def load(file_path: str='mnist.pkl.gz') -> Tuple[SharedDataset, SharedDataset, SharedDataset]:

    with gzip.open(file_path, 'rb') as file:

        try:

            train, validation, test = pickle.load(file, encoding='latin1')

        except:

            train, validation, test = pickle.load(file)

    training_vectors, training_labels = train
    validation_vectors, validation_labels = validation
    test_vectors, test_labels = test

    training_set = SharedDataset(training_vectors, training_labels)
    validation_set = SharedDataset(validation_vectors, validation_labels)
    test_set = SharedDataset(test_vectors, test_labels)

    return training_set, validation_set, test_set