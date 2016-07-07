import pickle
from theano import tensor as T
from theano.compile.function_module import Function

from initialization import zero_initialize


class GeneralizedLinearModel:

    W = T.dmatrix('W')
    b = T.dvector('b')

    x = T.dmatrix('x')
    y = T.ivector('y')

    def __init__(self,
                 vector_dim: int,
                 num_classes: int,
                 link_function: Function):

        self.vector_dim = vector_dim
        self.num_classes = num_classes
        self.link_function = link_function

        self.W_shape = (vector_dim, num_classes)
        self.b_shape = (num_classes,)

    @property
    def linear_projection(self):

        return T.dot(self.x, self.W) + self.b

    @property
    def response(self):

        return self.link_function(self.linear_projection)

    @property
    def prediction(self):

        return T.argmax(self.response, axis=1)

    def zero_initialize_weights(self):

        self.W = zero_initialize(self.W_shape, 'W')

    def zero_initialize_bias(self):

        self.b = zero_initialize(self.b_shape, 'b')


def load(file_path: str) -> GeneralizedLinearModel:

    return pickle.load(file_path, 'rb')


def save(model: GeneralizedLinearModel,
         target_path: str):

    with open(target_path, 'wb') as file:

        pickle._dump(model, file)