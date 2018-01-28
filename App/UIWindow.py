from PyQt5.QtWidgets import *


class UIWindow(object):
    def setupUI(self, MyApp):
        self.title = 'Convoultional Neural Network'
        self.left = 10
        self.top = 30
        self.width = 640
        self.height = 480
        MyApp.setWindowTitle(self.title)
        MyApp.setGeometry(self.left, self.top, self.width, self.height)
        self.centralwidget = QWidget(MyApp)
        MyApp.setCentralWidget(self.centralwidget)