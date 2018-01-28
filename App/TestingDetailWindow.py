from PyQt5.QtWidgets import QMainWindow

from App.TestingDetailWidget import TestingDetailWidget


class TestingDetailWindow(QMainWindow):
    def __init__(self):
        super().__init__()

    def setup_ui(self, MyApp, citra, true_label, dataset, file_bobot, model):
        self.setGeometry(200, 100, 800, 450)
        # MyApp.setFixedSize(600, 450)
        self.setWindowTitle("Testing Detail- Convolutional Neural Network")
        self.centralwidget = TestingDetailWidget(MyApp, citra, true_label, dataset, file_bobot, model)
        self.setCentralWidget(self.centralwidget)
