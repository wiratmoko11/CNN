import pickle as cPickle
import numpy as np

class Datasets():
    def __init__(self, datasets_folder):
        self.datasets_folder = datasets_folder
        
    def load_data_cifar10(self, batch):
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

    # Mengubah format matrix (3, 32, 32) menjadi format citra (32, 32, 3)
    def to_channel_last(self, data, shape):
        img = np.ndarray(shape=shape).astype(np.uint8)
        for i in range(3):
            img[:, :, i] = data[i, :, :]
        return img