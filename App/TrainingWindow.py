from PyQt5 import QtCore

from PyQt5.QtWidgets import *

from App.KurvaWidget import KurvaWidget
from App.TrainingThread import TrainingThread
from App.TrainingWidget import TrainingWidget


class TrainingWindow(QMainWindow):
    def __init__(self, MyApp, dataset, model):
        super().__init__()
        self.MyApp = MyApp
        self.model = model
        self.setup_ui(dataset)

    def setup_ui(self, dataset):
        self.setGeometry(50, 50, 400, 450)
        self.setFixedSize(600, 700)
        self.setWindowTitle("Training - Convolutional Neural Network")
        self.centralwidget = TrainingWidget(self, dataset=dataset, model=self.model)
        self.setCentralWidget(self.centralwidget)

    def closeEvent(self, event):
        self.MyApp.train_thread.terminate()
        print("Thread Terminated")








