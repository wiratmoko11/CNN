import sys
from PyQt5.QtCore import QThread
import theano.tensor as T

from Training import Training


class TrainingThread(QThread):
    def __init__(self, training_widget):
        QThread.__init__(self)
        print("Thread Call")
        self.training_widget = training_widget
        # self.sig.connect(self.training_widget.refresh_layout)
    def __del__(self):
        self.wait()

    def training(self):
        """

        :return:
        """

    def run(self):
        print("Start Training")
        training_x, training_y, testing_x, testing_y = self.training_widget.dataset.load_data_cifar10(batch=5)
        y_x = T.argmax(self.training_widget.pyx, axis=1)
        training = Training(x=self.training_widget.x, y=self.training_widget.y, params=self.training_widget.params, pyx=self.training_widget.pyx, y_x=y_x, is_visual=False)
        training.training(training_x, training_y, testing_x, testing_y, self.training_widget.vbox_visual)
        # self.training_widget.start_training()



