from PyQt5.QtWidgets import *

from App.TestingWidget import TestingWidget


class TestingWindow(object):
    def setup_ui(self, MyApp, dataset, dataset_folder):
        MyApp.setGeometry(200, 100, 800, 450)
        # MyApp.setFixedSize(600, 450)
        MyApp.setWindowTitle("Validating - Convolutional Neural Network")
        self.centralwidget = TestingWidget(MyApp, dataset, dataset_folder)
        MyApp.setCentralWidget(self.centralwidget)
