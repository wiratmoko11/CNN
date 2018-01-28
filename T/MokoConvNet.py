import numpy
import pandas
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

import matplotlib.pyplot as plot

from Dataset import Datasets
from MyPlotter import MyPlotter


class MokoConvNet(object):
    def __init__(self):
        print("Init")
    # def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2,2)):
    #     assert image_shape[1] == filter_shape[1]
    #
    #     self.input = input
    #     fan_in = numpy.prod(filter_shape[1:])
    #
    #     fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:])) // numpy.prod(poolsize)
    #
    #     W_bound = numpy.sqrt(6 / (fan_in + fan_out))
    #
    #     self.W = theano.shared(
    #         numpy.asarray(
    #             rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
    #             dtype=theano.config.floatX
    #         ),
    #         borrow = True
    #     )
    #
    #     b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
    #     self.b = theano.shared(value=b_values, borrow=True)
    #
    #     conv_out = conv2d(
    #         input=input,
    #         filters=self.W,
    #         filter_shape=filter_shape,
    #         input_shape=image_shape
    #     )
    #
    #     pooled_out = pool.pool_2d(
    #         input=conv_out,
    #         ds=poolsize,
    #         ignore_border=True
    #     )
    #
    #     self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
    #
    #     self.params = [self.W, self.b]
    #
    #     self.input = input
    #
    #     x = T.tensor4()
    #     y = T.matrix()
    #
    #     cost = T.mean(T.nnet.categorical_crossentropy)
    #
    #     self.train = theano.function(inputs=[x, y], outputs=cost)

    def model(self, x, w, w2, w3, w_output):
        X = T.tensor4()
        layer_0 = x
        layer_1 = self.relu(conv2d(input=layer_0, filters=w))
        # print(layer_1)

        # visual_layer1(x)
        layer_2 = pool.pool_2d(input=layer_1, ws=(3, 3), ignore_border=True)
        layer_3 = self.relu(conv2d(input=layer_2, filters=w2))
        layer_4 = pool.pool_2d(input=layer_3, ws=(2, 2),  ignore_border=True)

        layer_5 = T.flatten(layer_4, outdim=2)
        # print(layer_6)
        layer_6 = self.relu(T.dot(layer_5, w3))

        pyx = self.softmax(T.dot(layer_6, w_output))

        return layer_0, layer_1, layer_2, pyx

    def floatX(self, x):
        return numpy.asarray(x, dtype=theano.config.floatX)

    # test_x iku data citra test_y iku data label
    def accuracy(self, test_x, test_y):
        print("Accuracy")

    def plot(self):
        print("Plot")

    def relu(self, x):
        return T.maximum(x, 0.)

    def init_kernel(self, shape):
        return theano.shared(self.floatX(numpy.random.randn(*shape) * 0.01))

    def softmax(self, x):
        e_x = T.exp(x - x.max(axis=1).dimshuffle(0, 'x'))
        return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

    def show_kernel(self, kernel):
        plot.imshow(kernel, cmap=plot.cm.binary)
        plot.show()

    def training(self):

        batch_size = 1
        label_test = numpy.argmax(cnn.testing_y, axis=1)
        # df = pandas.DataFrame(cnn.testing_y)
        # print(df)
        # exit(0)
        print(len(self.training_x[0:50]))
        # plot.imshow(self.training_x[100].reshape(32,32,3).astype(numpy.uint8))
        # plot.show()
        # exit(1)
        plot.ion()
        for i in range(10):
            print("epoch ", (i+1))
            for start in range(0, len(self.training_x), batch_size):
                x_batch = self.training_x[start:start + batch_size]
                y_batch = self.training_y[start:start + batch_size]

                plot.subplot(3, 8, 1); plot.imshow(datasets.to_RGB(x_batch[0], (32, 32, 3)))
                # print(x_batch.shape)
                # plot.imshow(x_batch[0].reshape(32, 32, 3))
                # plot.gray()
                # print(self.visualize_conv(x_batch).shape)
                plot.gray()
                for layer1_index in range(4):
                    plot.subplot(3,8,layer1_index+9); plot.imshow(self.visual_layer1(x_batch)[0][layer1_index] / 255, cmap=plot.cm.binary)

                for layer2_index in range(4):
                    plot.subplot(3,8,layer2_index+17); plot.imshow(self.visual_layer2(x_batch)[0][layer2_index] / 255, cmap=plot.cm.binary)

                # plot.subplot(2,4,6); plot.imshow(self.visualize_conv(x_batch)[0][1] / 255, cmap=plot.cm.binary)
                # plot.subplot(2,4,7); plot.imshow(self.visualize_conv(x_batch)[0][2] / 255, cmap=plot.cm.binary)
                # plot.subplot(2,4,8); plot.imshow(self.visualize_conv(x_batch)[0][3] / 255, cmap=plot.cm.binary)
                plot.pause(0.5)

                plot.show()
                # self.cost = self.train(x_batch, y_batch)
                # print("Cost ", self.cost)
                # print(self.train(x_batch, y_batch))

            print("Prediction")
            prediction_test = cnn.predict(cnn.testing_x[0:10])
            print(prediction_test)
            accuracy = numpy.mean(prediction_test == label_test[0:10])
            print("Accuracy = ", accuracy)
            print("")




    def RMSprop(self, cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - lr * g))
        return updates


    def check_param(self, i, node, fn):
        ""
        # print("Input", [input for input in fn.inputs])
        # print("Output", [output for output in fn.outputs])

    def most_square_shape(self, num_blocks, blockshape=(1,1)):
        x, y = blockshape
        num_x = numpy.ceil(numpy.sqrt(num_blocks * y / float(x))).astype(numpy.int8)
        num_y = numpy.ceil(num_blocks / num_x).astype(numpy.int8)
        return (num_x, num_y)

    def visualize_grid(self, chunk, range=None):
        if chunk.ndim == 4 and chunk.shape[1] == 1:
            # this is a chunk with one input channel, select it
            chunk = chunk[:, 0]

        if chunk.ndim != 3:
            raise RuntimeError("Only 3D tensors or 4D tensors with one input channel are supported as input, input dimensionality is %d" % chunk.ndim)

        if range is None:
            range = chunk.min(), chunk.max()

        vmin, vmax = range

        patch_size = chunk.shape[1:]
        num_x, num_y = self.most_square_shape(chunk.shape[0], patch_size)

        #pad with zeros so that the number of filters equals num_x * num_y
        chunk_padded = numpy.zeros((num_x * num_y,) + patch_size)
        chunk_padded[:chunk.shape[0]] = chunk

        chunk_split = chunk_padded.reshape(num_x, num_y, patch_size[0], patch_size[1])
        chunk_with_border = numpy.ones((num_x, num_y, patch_size[0] + 1, patch_size[1] + 1)) * vmax
        chunk_with_border[:, :, :patch_size[0], :patch_size[1]] = chunk_split

        grid = chunk_with_border.transpose(0, 2, 1, 3).reshape(num_x * (patch_size[0] + 1), num_y * (patch_size[1] + 1))
        grid_with_left_border = numpy.ones((num_x * (patch_size[0] + 1) + 1, num_y * (patch_size[1] + 1) + 1)) * vmax
        grid_with_left_border[1:, 1:] = grid

        plot.imshow(grid_with_left_border, interpolation='nearest', cmap=plot.cm.binary, vmin=vmin, vmax=vmax)
        plot.show()

class HiddenLayer():
    def __init__(self):
        print("Hidden Layer")

if(__name__ == '__main__'):

    # theano.config.exce

    cnn = MokoConvNet()
    plotter = MyPlotter()
    w = cnn.init_kernel((4, 3, 3, 3))
    w2 = cnn.init_kernel((8, 4, 3, 3))
    # plotter.visual_weight3(w2.get_value())
    # print(w.get_value().shape[0])
    # exit()
    # cnn.visualize_grid(w.get_value())
    # exit()

    w3 = cnn.init_kernel((8 * 4 * 4, 100))
    w_output = cnn.init_kernel((100, 10))

    datasets = Datasets(datasets_folder="../Sample/cifar-10-batches-py/")

    x = T.tensor4()
    y = T.matrix()

    cnn.layer_0, cnn.layer_1, cnn.layer_2, pyx = cnn.model(x, w, w2, w3, w_output)
    cnn.visual_layer1 = theano.function(inputs=[x], outputs=cnn.layer_1, allow_input_downcast=True)
    cnn.visual_layer2 = theano.function(inputs=[x], outputs=cnn.layer_2, allow_input_downcast=True)
    # cnn.visualize_conv = theano.function(inputs=[x], outputs=cnn.layer_2, allow_input_downcast=True)
    y_x = T.argmax(pyx, axis=1)
    # print("pyx ", pyx)
    """
    Cost menggunakan cross entropy H(p, x) = -Sigma_x p(x)log(q(x),
    untuk p = nilai sebenarnya, q = nilai = predictive (nilai hasil fungsi)
    """
    cnn.cost = T.mean(T.nnet.categorical_crossentropy(pyx, y))

    params = [w, w2, w3, w_output]
    cnn.updates = cnn.RMSprop(cnn.cost, params, 0.001, 0.9, 1e-6)

    # Ambil Bobot
    # print(w.eval()[0])
    cnn.show_kernel(w.eval()[0])

    exit()

    cnn.train = theano.function(inputs=[x, y], outputs=cnn.cost, updates=cnn.updates, allow_input_downcast=True, mode=theano.compile.MonitorMode(post_func=cnn.check_param))
    cnn.predict = theano.function(inputs=[x], outputs=y_x)
    cnn.data_training = datasets.loadDatasets()
    # print(cnn.data_training)
    cnn.training_x = cnn.data_training['data']
    print()
    # plot.imshow(datasets.to_RGB(cnn.training_x.reshape(10000, 3, 32 ,32))[0])
    # plot.show()
    # print(cnn.training_x.shape[0])
    # data_training_a = numpy.zeros(shape=(10000, 3, 1024), dtype=float)
    data_training_b = numpy.zeros(shape=(10000, 3, 32, 32), dtype=float)
    for i in range(cnn.training_x.shape[0]):
        data_training_b[i] = cnn.training_x[i].reshape(3, 32, 32)

    plotter.visual_layer(data_training_b, datasets)

    exit()
    # print(cnn.training_x[0:10])
    # print(cnn.training_x[0:10].shape)
    cnn.training_x = data_training_b
    cnn.training_y = cnn.data_training['labels']

    cnn.testing_x, cnn.testing_y = datasets.loadTest()
    cnn.testing_x = numpy.reshape(cnn.testing_x, newshape=(10000, 3, 32, 32))
    cnn.testing_y = numpy.eye(10)[numpy.array(cnn.testing_y)]

    label_training = numpy.eye(10)[numpy.array(cnn.training_y)]

    cnn.training_y = label_training

    # print(label_training)

    # label_training = numpy.zeros(shape=(1, 10000))
    # label_training[0] = cnn.training_y
    # cnn.training_y = numpy.reshape(label_training, newshape=(10000, 1))
    # print(label_training[0][10])
    # cnn.training_y = label_training
    cnn.training()