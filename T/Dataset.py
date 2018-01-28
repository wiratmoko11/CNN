import pickle as cPickle
import pandas
import numpy as np
import matplotlib.pyplot as matplot
class Datasets(object):
    def __init__(self, datasets_folder):
        self.datasets_folder = datasets_folder
        print("Datasets")
        for i in range (1):
            with open(datasets_folder+'data_batch_'+str(i+1), 'rb') as data_training:

                training_dict = cPickle.load(data_training, encoding="latin1")
                print(training_dict['labels'])
                df = pandas.DataFrame(training_dict['labels'])
                print(df)
                # matplot.imshow(training_dict['data'])
                # print(training_dict['data'][1][1])
                image = np.reshape(training_dict['data'][87], newshape=(3, 1024))
                # print(np.reshape(image[0],newshape=(32, 32)))
                image_rgb = np.zeros(shape=(3, 32, 32))
                image_rgb[0] = np.reshape(image[0], newshape=(32, 32))
                image_rgb[1] = np.reshape(image[1] ,newshape=(32, 32))
                image_rgb[2] = np.reshape(image[2], newshape=(32, 32))
                # df2 = pandas.DataFrame(image_rgb[0])
                # print(df2)
                # print(image_rgb.shape)
                matplot.imshow(self.to_RGB(image_rgb, (32, 32, 3)))
                #matplot.show()
                # print()
        self.datasets = training_dict
        # print(self.datasets)


    # Mengubah format matrix (3, 32, 32) menjadi format citra (32, 32, 3)
    def to_RGB(self, data, shape):
        img = np.ndarray(shape=shape).astype(np.uint8)
        for i in range(3):
            img[:, :, i] = data[i, :, :]
        return img

    def to_channel_last(self, data, shape):
        img = np.ndarray(shape=shape).astype(np.uint8)
        for i in range(3):
            img[:, :, i] = data[i, :, :]
        return img


    def load_cifar_datasets(self, batch=5):
        training_x = []
        training_y = []
        for i in range(batch):
            with open(self.datasets_folder+'data_batch_'+str(i+1), 'rb') as data_training:
                training_dict = cPickle.load(data_training, encoding="latin1")
                training_x.append(np.reshape(training_dict['data'], newshape=(10000, 3, 32, 32)))
                # One Hot Encoding
                training_y.append(np.eye(10)[np.array(training_dict['labels'])])

        training_x = np.concatenate(training_x, axis=0)
        training_y = np.concatenate(training_y, axis=0)

        data_testing = open(self.datasets_folder+'test_batch', 'rb')
        testing_dict = cPickle.load(data_testing, encoding="latin1")
        testing_x = np.reshape(testing_dict['data'], newshape=(10000, 3, 32, 32))
        # One Hot Encoding
        testing_y = np.eye(10)[np.array(testing_dict['labels'])]

        return training_x, training_y, testing_x, testing_y


    def loadDatasets(self):
        return self.datasets

    def loadTest(self):
        ""
        data_test = open(self.datasets_folder+'test_batch', 'rb')
        test_dict = cPickle.load(data_test, encoding="latin1")
        x_test = test_dict['data']
        y_test = test_dict['labels']
        return x_test, y_test


    def showDatasets(self):
        print("Datasets")


if(__name__ =='__main__'):
    dataset = Datasets(datasets_folder="../Sample/cifar-10-batches-py/")

