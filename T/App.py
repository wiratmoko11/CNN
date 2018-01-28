import sys

import time
from PyQt5 import QtCore

import numpy
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ConvModel import ConvModel
from ConvNet import ConvNet
from Dataset import Datasets as Dataset, Datasets
from MyPlotter import MyPlotter
from Utils import Utils
import theano
import theano.tensor as T

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.widget = Widget(self)
        _widget = QWidget()
        _layout = QVBoxLayout(_widget)
        _layout.addWidget(self.widget)
        self.title = 'Convoultional Neural Network'
        self.left = 10
        self.top = 30
        self.width = 640
        self.height = 480
        self.initUI()
        self.setCentralWidget(_widget)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()

class Widget(QWidget):
    def __init__(self, parent):
        super(Widget, self).__init__(parent)

        self.__widget()
        self.__layout()


    def __widget(self):
        self.datasetLabel = QLabel('Conv')
        self.datasetFileLabel = QLabel('Dataset Belum Dipilih')
        self.pilihUserLabel = QLabel('Pilih User')

        self.figure = Figure()
        self.canvas = FigureCanvas(figure=self.figure)
        self.toolbar = NavigationToolbar(canvas=self.canvas, parent=self)

        self.figure_layer1 = Figure()
        self.canvas_layer1 = FigureCanvas(figure=self.figure_layer1)
        self.toolbar_layer1 = NavigationToolbar(canvas=self.canvas_layer1, parent=self)

        self.figure_layer2 = Figure()
        self.canvas_layer2 = FigureCanvas(figure=self.figure_layer2)
        self.toolbar_layer2 = NavigationToolbar(canvas=self.canvas_layer2, parent=self)

        self.figure_layer3 = Figure()
        self.canvas_layer3 = FigureCanvas(figure=self.figure_layer3)
        self.toolbar_layer3 = NavigationToolbar(canvas=self.canvas_layer3, parent=self)

        self.figure_layer4 = Figure()
        self.canvas_layer4 = FigureCanvas(figure=self.figure_layer4)
        self.toolbar_layer4 = NavigationToolbar(canvas=self.canvas_layer4, parent=self)

        self.figure_layer5 = Figure()
        self.canvas_layer5 = FigureCanvas(figure=self.figure_layer5)
        self.toolbar_layer5 = NavigationToolbar(canvas=self.canvas_layer5, parent=self)

        self.figure_layer6 = Figure()
        self.canvas_layer6 = FigureCanvas(figure=self.figure_layer6)
        self.toolbar_layer6 = NavigationToolbar(canvas=self.canvas_layer6, parent=self)

        self.figure_layer7 = Figure()
        self.canvas_layer7 = FigureCanvas(figure=self.figure_layer7)
        self.toolbar_layer7 = NavigationToolbar(canvas=self.canvas_layer7, parent=self)


    def __layout(self):
        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        # hbox.addWidget(self.toolbar)
        hbox.addWidget(self.canvas)

        hbox_layer1 = QHBoxLayout()
        hbox_layer1.addWidget(self.canvas_layer1)
        hbox_layer2 = QHBoxLayout()
        hbox_layer2.addWidget(self.canvas_layer2)
        hbox_layer3 = QHBoxLayout()
        hbox_layer3.addWidget(self.canvas_layer3)
        hbox_layer4 = QHBoxLayout()
        hbox_layer4.addWidget(self.canvas_layer4)
        hbox_layer5 = QHBoxLayout()
        hbox_layer5.addWidget(self.canvas_layer5)
        hbox_layer6 = QHBoxLayout()
        hbox_layer6.addWidget(self.canvas_layer6)
        hbox_layer7 = QHBoxLayout()
        hbox_layer7.addWidget(self.canvas_layer7)

        vbox.addLayout(hbox)
        vbox.addLayout(hbox_layer1)
        vbox.addLayout(hbox_layer2)
        vbox.addLayout(hbox_layer3)
        vbox.addLayout(hbox_layer4)
        vbox.addLayout(hbox_layer5)
        vbox.addLayout(hbox_layer6)
        vbox.addLayout(hbox_layer7)

        self.setLayout(vbox)

        self.plot()
        # self.train()




    def train(self):
        datasets = Datasets(datasets_folder="../Sample/cifar-10-batches-py/")
        model = ConvModel()
        plotter = MyPlotter()
        utils = Utils()

        x = T.tensor4()
        """
        y = true label
        """
        y = T.matrix()

        w1 = utils.init_kernel((4, 3, 3, 3))
        b1 = utils.init_kernel((4,))
        w2 = utils.init_kernel((8, 4, 3, 3))
        b2 = utils.init_kernel((8,))
        w3 = utils.init_kernel((200, 100))
        b3 = utils.init_kernel((100,))
        w_output = utils.init_kernel((100, 10))
        b_output = utils.init_kernel((10,))

        plotter.visual_weight3(w1.get_value())
        time.sleep(100)
        exit()

        params = [w1, b1, w2, b2, w3, b3, w_output, b_output]
        """
        pyx = prediction label of model
        """
        layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, pyx = model.model1(x, w1, b1, w2, b2, w3, b3, w_output, b_output)

        y_x = T.argmax(pyx, axis=1)

        visual_layer = []
        visual_layer.append(theano.function(inputs=[x], outputs=layer_0))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_1))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_2))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_3))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_4))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_5))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_6))
        visual_layer.append(theano.function(inputs=[x], outputs=pyx))

        convnet  = ConvNet(params=params, pyx=pyx, x=x, y=y, y_x=y_x)

        training_x, training_y, testing_x, testing_y = datasets.load_cifar_datasets(batch=1)

        figure_layer = []
        figure_layer.append(self.figure)
        figure_layer.append(self.figure_layer1)
        figure_layer.append(self.figure_layer2)
        figure_layer.append(self.figure_layer3)
        figure_layer.append(self.figure_layer4)
        figure_layer.append(self.figure_layer5)
        figure_layer.append(self.figure_layer6)
        figure_layer.append(self.figure_layer7)

        canvas = []
        canvas.append(self.canvas)
        canvas.append(self.canvas_layer1)
        canvas.append(self.canvas_layer2)
        canvas.append(self.canvas_layer3)
        canvas.append(self.canvas_layer4)
        canvas.append(self.canvas_layer5)
        canvas.append(self.canvas_layer6)
        canvas.append(self.canvas_layer7)

        convnet.training(training_x[0:100], training_y[0:100], testing_x[0:100], testing_y[0:100], visual_layer, figure_layer, canvas, datasets)

    def train_model_2(self):
        datasets = Datasets(datasets_folder="../Sample/cifar-10-batches-py/")
        model = ConvModel()
        plotter = MyPlotter()
        utils = Utils()

        x = T.tensor4()
        """
        y = true label
        """
        y = T.matrix()

        w1 = utils.init_kernel((16, 3, 5, 5))
        b1 = utils.init_kernel((16,))
        w2 = utils.init_kernel((20, 16, 5, 5))
        b2 = utils.init_kernel((20,))
        w3 = utils.init_kernel((20, 20, 5, 5))
        b3 = utils.init_kernel((20,))
        w_output = utils.init_kernel((320, 10))
        b_output = utils.init_kernel((10,))
        plotter.visual_weight3(w1.get_value());
        time.sleep(10)
        exit()
        # plotter.visual_weight()

        params = [w1, w2, w3, w_output]
        """
        pyx = prediction label of model
        """
        layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, pyx = model.model3(x, w1, w2, w3, w_output)

        y_x = T.argmax(pyx, axis=1)

        visual_layer = []
        visual_layer.append(theano.function(inputs=[x], outputs=layer_0))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_1))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_2))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_3))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_4))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_5))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_6))
        visual_layer.append(theano.function(inputs=[x], outputs=pyx))

        convnet  = ConvNet(params=params, pyx=pyx, x=x, y=y, y_x=y_x)

        training_x, training_y, testing_x, testing_y = datasets.load_cifar_datasets(batch=1)

        figure_layer = []
        figure_layer.append(self.figure)
        figure_layer.append(self.figure_layer1)
        figure_layer.append(self.figure_layer2)
        figure_layer.append(self.figure_layer3)
        figure_layer.append(self.figure_layer4)
        figure_layer.append(self.figure_layer5)
        figure_layer.append(self.figure_layer6)
        figure_layer.append(self.figure_layer7)

        canvas = []
        canvas.append(self.canvas)
        canvas.append(self.canvas_layer1)
        canvas.append(self.canvas_layer2)
        canvas.append(self.canvas_layer3)
        canvas.append(self.canvas_layer4)
        canvas.append(self.canvas_layer5)
        canvas.append(self.canvas_layer6)
        canvas.append(self.canvas_layer7)

        convnet.training(training_x[0:100], training_y[0:100], testing_x[0:100], testing_y[0:100], visual_layer, figure_layer, canvas, datasets)

    def train_model_3(self):
        datasets = Datasets(datasets_folder="../Sample/cifar-10-batches-py/")
        model = ConvModel()
        # plotter = MyPlotter()
        utils = Utils()

        x = T.tensor4()
        """
        y = true label
        """
        y = T.matrix()

        w1 = utils.init_kernel((16, 3, 5, 5))
        b1 = utils.init_kernel((16,))
        w2 = utils.init_kernel((20, 16, 5, 5))
        b2 = utils.init_kernel((20,))
        w3 = utils.init_kernel((1280, 128))
        b3 = utils.init_kernel((128,))
        w_output = utils.init_kernel((128, 10))
        b_output = utils.init_kernel((10,))

        # plotter.visual_weight()

        params = [w1, b1, w2, b2, w3, b3, w_output, b_output]
        """
        pyx = prediction label of model
        """
        layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, pyx = model.model3(x, w1, w2, w3, w_output)

        y_x = T.argmax(pyx, axis=1)

        visual_layer = []
        visual_layer.append(theano.function(inputs=[x], outputs=layer_0))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_1))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_2))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_3))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_4))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_5))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_6))
        visual_layer.append(theano.function(inputs=[x], outputs=pyx))

        convnet  = ConvNet(params=params, pyx=pyx, x=x, y=y, y_x=y_x)

        training_x, training_y, testing_x, testing_y = datasets.load_cifar_datasets(batch=1)

        figure_layer = []
        figure_layer.append(self.figure)
        figure_layer.append(self.figure_layer1)
        figure_layer.append(self.figure_layer2)
        figure_layer.append(self.figure_layer3)
        figure_layer.append(self.figure_layer4)
        figure_layer.append(self.figure_layer5)
        figure_layer.append(self.figure_layer6)
        figure_layer.append(self.figure_layer7)

        canvas = []
        canvas.append(self.canvas)
        canvas.append(self.canvas_layer1)
        canvas.append(self.canvas_layer2)
        canvas.append(self.canvas_layer3)
        canvas.append(self.canvas_layer4)
        canvas.append(self.canvas_layer5)
        canvas.append(self.canvas_layer6)
        canvas.append(self.canvas_layer7)

        convnet.training(training_x[0:100], training_y[0:100], testing_x[0:100], testing_y[0:100], visual_layer, figure_layer, canvas, datasets)

    def train_model_4(self):
        datasets = Datasets(datasets_folder="../Sample/cifar-10-batches-py/")
        model = ConvModel()
        # plotter = MyPlotter()
        utils = Utils()

        x = T.tensor4()
        """
        y = true label
        """
        y = T.matrix()

        w1 = utils.init_kernel((16, 3, 5, 5))
        b1 = utils.init_kernel((16,))
        w2 = utils.init_kernel((20, 16, 5, 5))
        w3 = utils.init_kernel((20, 20, 5, 5))
        w4 = utils.init_kernel((20, 20, 5, 5))
        w_output = utils.init_kernel((80, 10))

        # plotter.visual_weight()

        params = [w1, w2, w3, w_output]
        """
        pyx = prediction label of model
        """
        layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7, layer_8, layer_9a, pyx = model.model4(x, w1, w2, w3, w4, w_output)

        y_x = T.argmax(pyx, axis=1)

        visual_layer = []
        visual_layer.append(theano.function(inputs=[x], outputs=layer_0))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_1))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_2))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_3))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_4))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_5))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_6))
        visual_layer.append(theano.function(inputs=[x], outputs=pyx))

        convnet  = ConvNet(params=params, pyx=pyx, x=x, y=y, y_x=y_x)

        training_x, training_y, testing_x, testing_y = datasets.load_cifar_datasets(batch=1)

        figure_layer = []
        figure_layer.append(self.figure)
        figure_layer.append(self.figure_layer1)
        figure_layer.append(self.figure_layer2)
        figure_layer.append(self.figure_layer3)
        figure_layer.append(self.figure_layer4)
        figure_layer.append(self.figure_layer5)
        figure_layer.append(self.figure_layer6)
        figure_layer.append(self.figure_layer7)

        canvas = []
        canvas.append(self.canvas)
        canvas.append(self.canvas_layer1)
        canvas.append(self.canvas_layer2)
        canvas.append(self.canvas_layer3)
        canvas.append(self.canvas_layer4)
        canvas.append(self.canvas_layer5)
        canvas.append(self.canvas_layer6)
        canvas.append(self.canvas_layer7)

        convnet.training(training_x[0:100], training_y[0:100], testing_x[0:100], testing_y[0:100], visual_layer, figure_layer, canvas, datasets)


    def plot(self):
        dataset = Dataset(datasets_folder="../Sample/cifar-10-batches-py/")

        training_x, training_y, testing_x, testing_y = dataset.load_cifar_datasets(batch=1)

        self.canvas.draw()

class TrainThread(QtCore.QThread):
    sig = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        self.myapp = App()
        super(TrainThread, self).__init__(parent)
        # Connect signal to the desired function
        self.sig.connect(self.myapp.widget.plot)

    def run(self):
        self.myapp.widget.train()
        # print("asdasd")
        # while True:
        #     print("aasdsd")
            # val = sysInfo.getCpu()

            # Emit the signal
            # self.sig.emit(val)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    train_thread = TrainThread()
    ex = App()
    train_thread.start()
    #ex.widget.train()
    sys.exit(app.exec_())
