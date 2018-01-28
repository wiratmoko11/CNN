from PyQt5.QtWidgets import QApplication
import sys
from App import App
from ConvModel import ConvModel
from ConvNet import ConvNet
from Dataset import Datasets
from Utils import Utils
import theano.tensor as T
import theano

if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())



