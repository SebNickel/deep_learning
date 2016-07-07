from numpy import ndarray
import pickle


class Dataset:

    def __init__(self,
                 vectors: ndarray,
                 labels: ndarray):

        self.vectors = vectors
        self.labels = labels


def save(dataset: Dataset,
         file_path: str):

    with open(file_path, 'wb') as file:

        pickle._dump(dataset, file)


def load(file_path: str) -> Dataset:

    with open(file_path, 'rb') as file:

        dataset = pickle.load(file)

    return dataset
