from datasets import Dataset


def batch_iterator(dataset: Dataset,
                   batch_size: int):

    num_batches = dataset.vectors.shape[0] // batch_size

    for batch_index in range(num_batches):

        x_batch = dataset.vectors[batch_index * batch_size: (batch_index + 1) * batch_size]
        y_batch = dataset.labels[batch_index * batch_size: (batch_index + 1) * batch_size]

        yield x_batch, y_batch