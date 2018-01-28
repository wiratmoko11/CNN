import sys
import numpy as np
from theano import tensor as T
from Dataset import Datasets
from MokoConvNet import MokoConvNet, HiddenLayer


class Cnn1(object):
    def __init__(self, learning_rate=0.1, epoch = 10, datasest="cifar10", num_kernel=[20, 50], batch_size=10000):
        self.x = T.matrix(name="x")
        self.y = T.ivector(name="y")
        self.batch_size = batch_size
        self.num_kernel = num_kernel
        self.rng = 0

        self.build_model()

    def build_model(self):
        layer0_input = self.x.reshape((self.batch_size, 1, 32, 32))

        layer0 = MokoConvNet(
            rng=self.rng,
            input=layer0_input,
            image_shape=(self.batch_size, 1, 32, 32),
            filter_shape=(self.num_kernel[0], 1, 5, 5),
            poolsize=(2,2)
        )

        layer1 = MokoConvNet(
            self.rng,
            input = layer0.output,
            image_shape=(self.batch_size, self.num_kernel[0], 12, 12),
            filter_shape=(self.num_kernel[1], self.num_kernel[0], 5, 5)
        )

        layer2_input = layer1.output.flatten(2)

        layer2 = HiddenLayer(
            rng=self.rng,
            input=layer0_input,
            image_shape=(self.batch_size, 1, 32, 32),
            filter_shape=(self.num_kernel[0], 1, 5, 5),
            poolsize=(2,2)
        )

    def evaluate(self):
        print("evaluating")
        covnet = MokoConvNet(rng=self.rng)



if __name__ == '__main__':
    datasets = Datasets(datasets_folder="../Sample/cifar-10-batches-py/")
    cnn = Cnn1(sys.argv, learning_rate=0.1, epoch=1, datasest="cifar10", batch_size=10000)

    cnn.rng = np.random.RandomState(23455)
    # print(cnn.rng.rand())

    # cnn.evaluate()