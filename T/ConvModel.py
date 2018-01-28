from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
import theano.tensor as T
from Utils import  Utils


class ConvModel():
    utils = Utils()

    def model1(self, x, w1, b1, w2, b2, w3, b3, w_output, b_output):
        print("Model 1")
        layer_0 = x
        layer_1 = self.utils.relu(conv2d(input=layer_0, filters=w1, border_mode=1) + b1.dimshuffle('x', 0, 'x', 'x'))
        layer_2 = pool.pool_2d(input=layer_1, ws=(3, 3), ignore_border=True)
        layer_3 = self.utils.relu(conv2d(input=layer_2, filters=w2, border_mode=1) + b2.dimshuffle('x', 0, 'x', 'x'))
        layer_4 = pool.pool_2d(input=layer_3, ws=(2, 2),  ignore_border=True)
        layer_5 = T.flatten(layer_4, outdim=2)
        layer_6 = self.utils.relu(T.dot(layer_5, w3) + b3)
        pyx = self.utils.softmax(T.dot(layer_6, w_output) + b_output)

        return layer_0, layer_1, layer_2,  layer_3, layer_4, layer_5, layer_6, pyx

    def model2(self, x, w1, b1, w2, b2, w3, b3, w_output, b_output):
        print("Model 2")
        layer_0 = x
        layer_1 = self.utils.relu(conv2d(input=layer_0, filters=w1, border_mode=2) + b1.dimshuffle('x', 0, 'x', 'x'))
        layer_2 = pool.pool_2d(input=layer_1, ws=(2, 2), ignore_border=True)
        layer_3 = self.utils.relu(conv2d(input=layer_2, filters=w2, border_mode=2) + b2.dimshuffle('x', 0, 'x', 'x'))
        layer_4 = pool.pool_2d(input=layer_3, ws=(2,2), ignore_border=True)
        layer_5 = self.utils.relu(conv2d(input=layer_4, filters=w3, border_mode=2) + b3.dimshuffle('x', 0, 'x', 'x'))
        layer_6 = pool.pool_2d(input=layer_5, ws=(2,2), ignore_border=True)
        layer_7a = T.flatten(layer_6, outdim=2)
        pyx = self.utils.softmax(T.dot(layer_7a, w_output) + b_output)
        return layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7a, pyx



    def model3(self, x, w1, b1, w2, b2, w3, b3, w_output, b_output):
        print("Model 3")
        layer_0 = x
        layer_1 = self.utils.relu(conv2d(input=layer_0, filters=w1, border_mode=2) + b1.dimshuffle('x', 0, 'x', 'x'))
        layer_2 = pool.pool_2d(input=layer_1, ws=(2, 2), ignore_border=True)
        layer_3 = self.utils.relu(conv2d(input=layer_2, filters=w2, border_mode=2) + b2.dimshuffle('x', 0, 'x', 'x') )
        layer_4 = pool.pool_2d(input=layer_3, ws=(2, 2),  ignore_border=True)
        layer_5 = T.flatten(layer_4, outdim=2)
        layer_6 = self.utils.relu(T.dot(layer_5, w3) + b3)
        pyx = self.utils.softmax(T.dot(layer_6, w_output) + b_output)

        return layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, pyx

    def model4(self, x, w1, w2, w3, w4, w_output):
        print("Model 4")
        layer_0 = x
        layer_1 = self.utils.relu(conv2d(input=layer_0, filters=w1, border_mode=2))
        layer_2 = pool.pool_2d(input=layer_1, ws=(2, 2), ignore_border=True)
        layer_3 = self.utils.relu(conv2d(input=layer_2, filters=w2, border_mode=2))
        layer_4 = pool.pool_2d(input=layer_3, ws=(2,2), ignore_border=True)
        layer_5 = self.utils.relu(conv2d(input=layer_4, filters=w3, border_mode=2))
        layer_6 = pool.pool_2d(input=layer_5, ws=(2,2), ignore_border=True)
        layer_7 = self.utils.relu(conv2d(input=layer_6, filters=w4, border_mode=2))
        layer_8 = pool.pool_2d(input=layer_7, ws=(2,2), ignore_border=True)
        layer_9a = T.flatten(layer_8, outdim=2)
        pyx = self.utils.softmax(T.dot(layer_9a, w_output))

        return layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7, layer_8, layer_9a, pyx