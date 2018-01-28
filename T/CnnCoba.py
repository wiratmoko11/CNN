from theano import tensor as T
import numpy
import theano
from theano.tensor.nnet import conv2d

from MokoConvNet import MokoConvNet


class CnnCoba():
    def __init__(self, batch_size, num_kernel):
        self.x = T.matrix(name="x")
        self.y = T.ivector(name="y")
        self.batch_size = batch_size
        self.num_kernel = num_kernel
        self.rng = 0

    def build_model(self):

        fan_in = numpy.prod(filter_shape[1:])

        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:])) // numpy.prod(poolsize)

        W_bound = numpy.sqrt(6 / (fan_in + fan_out))

        self.W = theano.shared(
            numpy.asarray(
                self.rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow = True
        )

        layer0_input = self.x.reshape((self.batch_size, 1, 32, 32))
        layer0 = conv2d(

        )

        index = T.lscalar()

        train_model = theano.function(
            [index],
            layer0
        )
