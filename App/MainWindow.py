from PyQt5.QtWidgets import *

from App.MainWidget import MainWidget
from App.TrainingThread import TrainingThread


class MainWindow(object):
    def setup_ui(self, MyApp, dataset, dataset_folder):
        self.dataset = dataset
        self.MyApp = MyApp
        self.MyApp.setGeometry(200, 100, 800, 450)
        # MyApp.setFixedSize(600, 450)
        self.MyApp.setWindowTitle("Convolutional Neural Network")
        self.centralwidget = MainWidget(self.MyApp, dataset, dataset_folder)
        self.MyApp.setCentralWidget(self.centralwidget)
        self.centralwidget.buttonTrain.clicked.connect(self.handle_train_button)

    def handle_train_button(self):
        # training_widget = TrainingWidget(dataset=self.dataset, model=self.centralwidget.model)
        print("Training")
        self.MyApp.start_ui_training()

