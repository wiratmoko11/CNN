import sys

import time
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import *

from App.MainWidget import MainWidget
from App.MainWindow import MainWindow
from App.TestingDetailWindow import TestingDetailWindow
from App.TestingWindow import TestingWindow
from App.TrainingThread import TrainingThread
from App.TrainingWidget import TrainingWidget
from App.TrainingWindow import TrainingWindow
from Datasets import Datasets
import theano.tensor as T

from Training import Training


class MyApp(QMainWindow):
    def __init__(self, parent=None):
        super(MyApp, self).__init__(parent)
        self.dataset_folder = "../../Sample/cifar-10-batches-py/"
        self.dataset = Datasets(datasets_folder=self.dataset_folder)
        self.ui_main = MainWindow()
        self.ui_training = TrainingWindow(self, self.dataset, "Model 1")
        self.ui_testing = TestingWindow()
        self.ui_testing_detail = TestingDetailWindow()


        self.start_ui_main()
        self.init_ui_component()

    def init_ui_component(self):
        exitAct = QAction(QIcon('../assets/power-button.png'), 'Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.triggered.connect(qApp.quit)

        trainingAct = QAction(QIcon('../assets/home.png'), 'Training', self)
        trainingAct.triggered.connect(self.start_ui_main)
        testingAct = QAction(QIcon('../assets/test.png'), 'Testing', self)
        testingAct.triggered.connect(self.start_ui_testing)

        q_toolbar = QToolBar()
        q_toolbar.addAction(trainingAct)
        q_toolbar.addAction(testingAct)
        q_toolbar.addAction(exitAct)
        self.toolbar = self.addToolBar(QtCore.Qt.LeftToolBarArea, q_toolbar)

    def start_ui_testing(self):
        self.ui_testing.setup_ui(self, self.dataset, self.dataset_folder)
        self.show()

    def start_ui_main(self):
        self.ui_main.setup_ui(self, self.dataset, self.dataset_folder)
        self.show()

    def start_ui_testing_detail(self):
        citra = self.ui_testing.centralwidget.citra
        true_label = self.ui_testing.centralwidget.true_label
        self.ui_testing_detail.setup_ui(self, citra, true_label, self.dataset, self.ui_testing.centralwidget.filename, self.ui_testing.centralwidget.model)
        self.ui_testing_detail.show()

    def start_ui_training(self):
        # train_app = TrainingWindow(self.dataset)
        epoch = self.ui_main.centralwidget.epochInput.value()
        self.ui_training = TrainingWindow(self, self.dataset, model=self.ui_main.centralwidget.model)
        self.ui_training.centralwidget.setLayout(self.ui_training.centralwidget.main_hbox)
        self.train_thread = TrainThread(self.ui_training.centralwidget, epoch)
        self.train_thread.start()
        # train_app.show()

        self.ui_training.show()

        """"""
        # self.ui_training
        # self.ui_training.setup_ui(self, self.dataset)
        # self.show()

        # training_thread = TrainingThread(self, self.dataset, self.ui_main.centralwidget.model)
        # training_thread.start()

        # training_thread.set_widget(training_widget=self.ui_training.centralwidget)
        # training_thread.start()

class TrainThread(QtCore.QThread):
    sig = QtCore.pyqtSignal(int)
    def __init__(self, training_widget, epoch, parent=None):
        super(TrainThread, self).__init__(parent)
        self.training_widget = training_widget
        self.epoch = epoch
    def __del__(self):
        self.wait()

    def run(self):
        print("RUN THREAD")
        epoch = self.epoch;
        training_x, training_y, testing_x, testing_y = self.training_widget.dataset.load_data_cifar10(batch=5)
        y_x = T.argmax(self.training_widget.pyx, axis=1)
        training = Training(x=self.training_widget.x, y=self.training_widget.y, params=self.training_widget.params, pyx=self.training_widget.pyx, y_x=y_x, is_visual=True)
        training.training(training_x, training_y, testing_x, testing_y, self.training_widget, dataset=self.training_widget.dataset, p_epochs=epoch)


if __name__  == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    # ex.show()
    app.exec_()


