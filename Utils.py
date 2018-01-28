import numpy
import theano
import theano.tensor as T


class Utils():
    """
    Fungsi ubah nilai ke type float
    """
    def floatX(self, x):
        return numpy.asarray(x, dtype=theano.config.floatX)
    """
    Fungsi create bobot
    """
    def init_kernel(self, shape):
        return theano.shared(self.floatX(numpy.random.randn(*shape) * 0.01))

    """
    Fungsi aktivasi ReLU
    """
    def relu(self, x):
        return T.maximum(x, 0.)

    """
    Fungsi Softmax
    """
    def softmax(self, x):
        e_x = T.exp(x - x.max(axis=1).dimshuffle(0, 'x'))
        return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')
