import time
from PyQt5 import QtCore

from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import theano.tensor as T
from PyQt5.QtCore import QThread
# from App.TrainingThread import TrainingThread
from App.TrainingThread import TrainingThread
from ConvModel import ConvModel
from Training import Training
from Utils import Utils
import theano


class TrainingWidget(QWidget):
    x = T.tensor4()
    y = T.matrix()
    def __init__(self, parent, dataset, model):
        super(TrainingWidget, self).__init__(parent)
        self.utils = Utils()
        self.parent = parent
        self.model = model
        self.dataset = dataset
        self._components()
        self.progress_bar_value = 0
        self.pilih_model()
        self.epoch = 1;
        # self._layout()
        # if(self.model == "Model 1"):
        #     self._component_model_1()
        #     self._layout_model_1()
        #     self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output = self.init_weight_model1()
        #     layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, self.pyx = conv_model.init_model_1(self.x, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output)
        #     self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output]
        #     self.visual_layer = self.init_visual_model_1(self.x, layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, self.pyx)
        # elif(self.model == "Model 2"):
        #     self._component_model_2()
        #     self._layout_model_2()
        #     self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output = self.init_weight_model2()
        #     layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7a, self.pyx = conv_model.init_model_2(self.x, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output)
        #     self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output]
        #     self.visual_layer = self.init_visual_model_2(self.x, layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7a, self.pyx)
        # elif(self.model == "Model 3"):
        #     self._component_model_3()
        #     self._layout_model_3()
        #     self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output = self.init_weight_model3()
        #     layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, self.pyx = conv_model.init_model_3(self.x, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output)
        #     self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output]
        #     self.visual_layer = self.init_visual_model_3(self.x, layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, self.pyx)

        # time.sleep(10)

    def pilih_model(self):
        conv_model = ConvModel()
        # self.setLayout()

        if(self.model == "Model 1"):
            self._component_model_1()
            self._layout_model_1()
            self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output = self.init_weight_model1()
            layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, self.pyx = conv_model.init_model_1(self.x, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output)
            self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output]
            self.visual_layer = self.init_visual_model_1(self.x, layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, self.pyx)
        elif(self.model == "Model 2"):
            self._component_model_2()
            self._layout_model_2()
            self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output = self.init_weight_model2()
            layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7a, self.pyx = conv_model.init_model_2(self.x, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output)
            self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output]
            self.visual_layer = self.init_visual_model_2(self.x, layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7a, self.pyx)
        elif(self.model == "Model 3"):
            self._component_model_3()
            self._layout_model_3()
            self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output = self.init_weight_model3()
            layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, self.pyx = conv_model.init_model_3(self.x, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output)
            self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output]
            self.visual_layer = self.init_visual_model_3(self.x, layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, self.pyx)



    def init_weight_model1(self):
        w1 = self.utils.init_kernel((4, 3, 3, 3))
        b1 = self.utils.init_kernel((4,))
        w2 = self.utils.init_kernel((8, 4, 3, 3))
        b2 = self.utils.init_kernel((8,))
        w3 = self.utils.init_kernel((200, 100))
        b3 = self.utils.init_kernel((100,))
        w_output = self.utils.init_kernel((100, 10))
        b_output = self.utils.init_kernel((10,))

        return w1, b1, w2, b2, w3, b3, w_output, b_output

    def init_weight_model2(self):
        w1 = self.utils.init_kernel((16, 3, 5, 5))
        b1 = self.utils.init_kernel((16,))
        w2 = self.utils.init_kernel((20, 16, 5, 5))
        b2 = self.utils.init_kernel((20,))
        w3 = self.utils.init_kernel((20, 20, 5, 5))
        b3 = self.utils.init_kernel((20,))
        w_output = self.utils.init_kernel((320, 10))
        b_output = self.utils.init_kernel((10,))

        return w1, b1, w2, b2, w3, b3, w_output, b_output

    def init_weight_model3(self):
        w1 = self.utils.init_kernel((16, 3, 5, 5))
        b1 = self.utils.init_kernel((16,))
        w2 = self.utils.init_kernel((20, 16, 5, 5))
        b2 = self.utils.init_kernel((20,))
        w3 = self.utils.init_kernel((1280, 128))
        b3 = self.utils.init_kernel((128,))
        w_output = self.utils.init_kernel((128, 10))
        b_output = self.utils.init_kernel((10,))

        return w1, b1, w2, b2, w3, b3, w_output, b_output

    def _components(self):
        self.button = QPushButton("Mulai")
        self.button.clicked.connect(self.act_start)
        self.progress_bar = QProgressBar(self)
        """

        :return:
        """

    def _component_model_1(self):
        self.input_label = QLabel("Input 32 x 32 x 3")
        self.conv1_label = QLabel("Conv 32 x 32 x 4")
        self.pool1_label = QLabel("Pooling 1 (3 x 3)")
        self.conv2_label = QLabel("Conv 32 x 32 x 16")
        self.w1_label = QLabel("Weight (3 x 3 x 4)")
        self.w2_label = QLabel("Weight (3 x 3 x 4)")
        self.pool2_label = QLabel("Pooling 2 (2 x 2)")
        self.fc1_label = QLabel("FC 1")
        self.fc2_label = QLabel("FC 2")
        self.fc_softmax_label = QLabel("Softmax")
        self.fc_output_label = QLabel("Output")

        self.figure_input = Figure(); self.canvas_input = FigureCanvas(self.figure_input)
        self.figure_conv1 = Figure(); self.canvas_conv1 = FigureCanvas(figure=self.figure_conv1)
        self.figure_w1 = Figure(); self.canvas_w1 = FigureCanvas(figure=self.figure_w1)
        self.figure_pooling1 = Figure(); self.canvas_pooling1 = FigureCanvas(self.figure_pooling1)

        self.figure_conv2 = Figure(); self.canvas_conv2 = FigureCanvas(figure=self.figure_conv2)
        self.figure_w2 = Figure(); self.canvas_w2 = FigureCanvas(figure=self.figure_w2)
        self.figure_pooling2 = Figure(); self.canvas_pooling2 = FigureCanvas(self.figure_pooling2)

        self.figure_fc1 = Figure(); self.canvas_fc1 = FigureCanvas(figure=self.figure_fc1)
        self.figure_fc2 = Figure(); self.canvas_fc2 = FigureCanvas(figure=self.figure_fc2)

        self.figure_softmax = Figure(); self.canvas_softmax = FigureCanvas(figure=self.figure_softmax)
        self.figure_output = Figure(); self.canvas_output = FigureCanvas(figure=self.figure_output)

    def _component_model_2(self):

        self.input_label = QLabel("Input 32 x 32 x 3")
        self.conv1_label = QLabel("Conv 32 x 32 x 16")
        self.pool1_label = QLabel("Pooling 1 (2 x 2)")
        self.conv2_label = QLabel("Conv 16 x 16 x 20")
        self.w1_label = QLabel("Weight 16 x (5 x 5 x 3)")
        self.w2_label = QLabel("Weight 20 x (5 x 5 x 16)")
        self.pool2_label = QLabel("Pooling 2 (2 x 2)")
        self.w3_label = QLabel("Weight 20 x (5 x 5 x 20)")
        self.conv3_label = QLabel("Conv 20 x 5 x 5 x 20")
        self.pool3_label = QLabel("Pooling 3 (2 x 2)")
        self.fc1_label = QLabel("FC 1")
        self.fc_softmax_label = QLabel("Softmax")
        self.fc_output_label = QLabel("Output")

        self.figure_input = Figure(); self.canvas_input = FigureCanvas(self.figure_input)
        self.figure_conv1 = Figure(); self.canvas_conv1 = FigureCanvas(figure=self.figure_conv1)
        self.figure_w1 = Figure(); self.canvas_w1 = FigureCanvas(figure=self.figure_w1)
        self.figure_pooling1 = Figure(); self.canvas_pooling1 = FigureCanvas(self.figure_pooling1)

        self.figure_conv2 = Figure(); self.canvas_conv2 = FigureCanvas(figure=self.figure_conv2)
        self.figure_w2 = Figure(); self.canvas_w2 = FigureCanvas(figure=self.figure_w2)
        self.figure_pooling2 = Figure(); self.canvas_pooling2 = FigureCanvas(self.figure_pooling2)

        self.figure_conv3 = Figure(); self.canvas_conv3 = FigureCanvas(figure=self.figure_conv3)
        self.figure_w3 = Figure(); self.canvas_w3 = FigureCanvas(figure=self.figure_w3)
        self.figure_pooling3 = Figure(); self.canvas_pooling3 = FigureCanvas(self.figure_pooling3)

        self.figure_fc1 = Figure(); self.canvas_fc1 = FigureCanvas(figure=self.figure_fc1)

        self.figure_softmax = Figure(); self.canvas_softmax = FigureCanvas(figure=self.figure_softmax)
        self.figure_output = Figure(); self.canvas_output = FigureCanvas(figure=self.figure_output)

    def _component_model_3(self):

        self.input_label = QLabel("Input 32 x 32 x 3")
        self.conv1_label = QLabel("Conv 32 x 32 x 16")
        self.pool1_label = QLabel("Pooling 1 (2 x 2)")
        self.conv2_label = QLabel("Conv 16 x 16 x 20")
        self.conv3_label = QLabel("Conv 8 x 8 x 20")
        self.w1_label = QLabel("Weight 16 x (5 x 5 x 3)")
        self.w2_label = QLabel("Weight 20 x (5 x 5 x 16)")
        self.w3_label = QLabel("Weight 20 x (5 x 5 x 20)")
        self.pool2_label = QLabel("Pooling 2 (2 x 2)")
        self.pool3_label = QLabel("Pooling 3 (2 x 2)")
        self.fc1_label = QLabel("FC 1")
        self.fc2_label = QLabel("FC 2")
        self.fc_softmax_label = QLabel("Softmax")
        self.fc_output_label = QLabel("Output")

        """"""
        self.figure_input = Figure(); self.canvas_input = FigureCanvas(self.figure_input)
        self.figure_conv1 = Figure(); self.canvas_conv1 = FigureCanvas(figure=self.figure_conv1)
        self.figure_w1 = Figure(); self.canvas_w1 = FigureCanvas(figure=self.figure_w1)
        self.figure_pooling1 = Figure(); self.canvas_pooling1 = FigureCanvas(self.figure_pooling1)

        self.figure_conv2 = Figure(); self.canvas_conv2 = FigureCanvas(figure=self.figure_conv2)
        self.figure_w2 = Figure(); self.canvas_w2 = FigureCanvas(figure=self.figure_w2)
        self.figure_pooling2 = Figure(); self.canvas_pooling2 = FigureCanvas(self.figure_pooling2)

        self.figure_conv3 = Figure(); self.canvas_conv3 = FigureCanvas(figure=self.figure_conv3)
        self.figure_w3 = Figure(); self.canvas_w3 = FigureCanvas(figure=self.figure_w3)
        self.figure_pooling3 = Figure(); self.canvas_pooling3 = FigureCanvas(self.figure_pooling3)

        self.figure_fc1 = Figure(); self.canvas_fc1 = FigureCanvas(figure=self.figure_fc1)
        self.figure_fc2 = Figure(); self.canvas_fc2 = FigureCanvas(figure=self.figure_fc2)

        self.figure_softmax = Figure(); self.canvas_softmax = FigureCanvas(figure=self.figure_softmax)
        self.figure_output = Figure(); self.canvas_output = FigureCanvas(figure=self.figure_output)

    def _layout_model_1(self):
        hbox_utama = QHBoxLayout()
        self.vbox_visual = QVBoxLayout()

        hbox_input = QHBoxLayout()
        hbox_input.addWidget(self.input_label)
        hbox_input.addWidget(self.canvas_input)

        hbox_w1 = QHBoxLayout()
        hbox_w1.addWidget(self.w1_label)
        hbox_w1.addWidget(self.canvas_w1)

        hbox_conv1 = QHBoxLayout()
        hbox_conv1.addWidget(self.conv1_label)
        hbox_conv1.addWidget(self.canvas_conv1)

        hbox_pool1 = QHBoxLayout()
        hbox_pool1.addWidget(self.pool1_label)
        hbox_pool1.addWidget(self.canvas_pooling1)

        hbox_w2 = QHBoxLayout()
        hbox_w2.addWidget(self.w2_label)
        hbox_w2.addWidget(self.canvas_w2)

        hbox_conv2 = QHBoxLayout()
        hbox_conv2.addWidget(self.conv2_label)
        hbox_conv2.addWidget(self.canvas_conv2)

        hbox_pool2 = QHBoxLayout()
        hbox_pool2.addWidget(self.pool2_label)
        hbox_pool2.addWidget(self.canvas_pooling2)


        hbox_softmax = QHBoxLayout()
        hbox_softmax.addWidget(self.fc_softmax_label)
        hbox_softmax.addWidget(self.canvas_softmax)

        hbox_fc1 = QHBoxLayout()
        hbox_fc1.addWidget(self.fc1_label)
        hbox_fc1.addWidget(self.canvas_fc1)

        hbox_fc2 = QHBoxLayout()
        hbox_fc2.addWidget(self.fc2_label)
        hbox_fc2.addWidget(self.canvas_fc2)

        hbox_output = QHBoxLayout()
        hbox_output.addWidget(self.fc_output_label)
        hbox_output.addWidget(self.canvas_output)

        self.vbox_visual.addLayout(hbox_input)
        self.vbox_visual.addLayout(hbox_w1)
        self.vbox_visual.addLayout(hbox_conv1)
        self.vbox_visual.addLayout(hbox_pool1)
        self.vbox_visual.addLayout(hbox_w2)
        self.vbox_visual.addLayout(hbox_conv2)
        self.vbox_visual.addLayout(hbox_pool2)
        self.vbox_visual.addLayout(hbox_fc1)
        self.vbox_visual.addLayout(hbox_fc2)
        self.vbox_visual.addLayout(hbox_softmax)


        self.vbox_visual.addWidget(self.progress_bar)
        hbox_utama.addLayout(self.vbox_visual)
        # self.setLayout(hbox_utama)

        self.main_hbox = hbox_utama

    def _layout_model_2(self):
        hbox_utama = QHBoxLayout()
        self.vbox_visual = QVBoxLayout()
        # input
        hbox_input = QHBoxLayout()
        hbox_input.addWidget(self.input_label)
        hbox_input.addWidget(self.canvas_input)
        # w1
        hbox_w1 = QHBoxLayout()
        hbox_w1.addWidget(self.w1_label)
        hbox_w1.addWidget(self.canvas_w1)
        # conv1
        hbox_conv1 = QHBoxLayout()
        hbox_conv1.addWidget(self.conv1_label)
        hbox_conv1.addWidget(self.canvas_conv1)
        #pool
        hbox_pool1 = QHBoxLayout()
        hbox_pool1.addWidget(self.pool1_label)
        hbox_pool1.addWidget(self.canvas_pooling1)
        # w2
        hbox_w2 = QHBoxLayout()
        hbox_w2.addWidget(self.w2_label)
        hbox_w2.addWidget(self.canvas_w2)
        #conv2
        hbox_conv2 = QHBoxLayout()
        hbox_conv2.addWidget(self.conv2_label)
        hbox_conv2.addWidget(self.canvas_conv2)
        #pool
        hbox_pool2 = QHBoxLayout()
        hbox_pool2.addWidget(self.pool2_label)
        hbox_pool2.addWidget(self.canvas_pooling2)
        # w3
        hbox_w3 = QHBoxLayout()
        hbox_w3.addWidget(self.w3_label)
        hbox_w3.addWidget(self.canvas_w3)
        #conv3
        hbox_conv3 = QHBoxLayout()
        hbox_conv3.addWidget(self.conv3_label)
        hbox_conv3.addWidget(self.canvas_conv3)
        #pool
        hbox_pool3 = QHBoxLayout()
        hbox_pool3.addWidget(self.pool3_label)
        hbox_pool3.addWidget(self.canvas_pooling3)
        #fc1
        hbox_fc1 = QHBoxLayout()
        hbox_fc1.addWidget(self.fc1_label)
        hbox_fc1.addWidget(self.canvas_fc1)
        #Softmax
        hbox_softmax = QHBoxLayout()
        hbox_softmax.addWidget(self.fc_softmax_label)
        hbox_softmax.addWidget(self.canvas_softmax)
        #output
        hbox_output = QHBoxLayout()
        hbox_output.addWidget(self.fc_output_label)
        hbox_output.addWidget(self.canvas_output)

        self.vbox_visual.addLayout(hbox_input)
        self.vbox_visual.addLayout(hbox_w1)
        self.vbox_visual.addLayout(hbox_conv1)
        self.vbox_visual.addLayout(hbox_pool1)
        self.vbox_visual.addLayout(hbox_w2)
        self.vbox_visual.addLayout(hbox_conv2)
        self.vbox_visual.addLayout(hbox_pool2)
        self.vbox_visual.addLayout(hbox_w3)
        self.vbox_visual.addLayout(hbox_conv3)
        self.vbox_visual.addLayout(hbox_pool3)
        self.vbox_visual.addLayout(hbox_fc1)
        self.vbox_visual.addLayout(hbox_softmax)

        hbox_utama.addLayout(self.vbox_visual)
        self.main_hbox = hbox_utama
        # self.setLayout(hbox_utama)

    def _layout_model_3(self):
        hbox_utama = QHBoxLayout()
        self.vbox_visual = QVBoxLayout()

        hbox_input = QHBoxLayout()
        hbox_input.addWidget(self.input_label)
        hbox_input.addWidget(self.canvas_input)

        hbox_w1 = QHBoxLayout()
        hbox_w1.addWidget(self.w1_label)
        hbox_w1.addWidget(self.canvas_w1)

        hbox_conv1 = QHBoxLayout()
        hbox_conv1.addWidget(self.conv1_label)
        hbox_conv1.addWidget(self.canvas_conv1)

        hbox_pool1 = QHBoxLayout()
        hbox_pool1.addWidget(self.pool1_label)
        hbox_pool1.addWidget(self.canvas_pooling1)

        hbox_w2 = QHBoxLayout()
        hbox_w2.addWidget(self.w2_label)
        hbox_w2.addWidget(self.canvas_w2)

        hbox_conv2 = QHBoxLayout()
        hbox_conv2.addWidget(self.conv2_label)
        hbox_conv2.addWidget(self.canvas_conv2)

        hbox_pool2 = QHBoxLayout()
        hbox_pool2.addWidget(self.pool2_label)
        hbox_pool2.addWidget(self.canvas_pooling2)

        hbox_softmax = QHBoxLayout()
        hbox_softmax.addWidget(self.fc_softmax_label)
        hbox_softmax.addWidget(self.canvas_softmax)

        hbox_fc1 = QHBoxLayout()
        hbox_fc1.addWidget(self.fc1_label)
        hbox_fc1.addWidget(self.canvas_fc1)

        hbox_fc2 = QHBoxLayout()
        hbox_fc2.addWidget(self.fc2_label)
        hbox_fc2.addWidget(self.canvas_fc2)

        hbox_output = QHBoxLayout()
        hbox_output.addWidget(self.fc_output_label)
        hbox_output.addWidget(self.canvas_output)

        self.vbox_visual.addLayout(hbox_input)
        self.vbox_visual.addLayout(hbox_w1)
        self.vbox_visual.addLayout(hbox_conv1)
        self.vbox_visual.addLayout(hbox_pool1)
        self.vbox_visual.addLayout(hbox_w2)
        self.vbox_visual.addLayout(hbox_conv2)
        self.vbox_visual.addLayout(hbox_pool2)
        self.vbox_visual.addLayout(hbox_fc1)
        self.vbox_visual.addLayout(hbox_fc2)
        self.vbox_visual.addLayout(hbox_softmax)

        hbox_utama.addLayout(self.vbox_visual)
        # self.setLayout(hbox_utama)
        self.main_hbox = hbox_utama

    def init_visual_model_1(self, x, layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, pyx):
        visual_layer = []
        visual_layer.append(theano.function(inputs=[x], outputs=layer_0))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_1))
        visual_layer.append(theano.shared(self.w1.get_value()))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_2))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_3))
        visual_layer.append(self.w2)
        visual_layer.append(theano.function(inputs=[x], outputs=layer_4))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_5))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_6))
        visual_layer.append(self.w3)
        visual_layer.append(theano.function(inputs=[x], outputs=pyx))
        visual_layer.append(self.w_output)

        return visual_layer

    def init_visual_model_2(self, x, layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7, pyx):
        visual_layer = []
        visual_layer.append(theano.function(inputs=[x], outputs=layer_0))
        #Conv 1
        visual_layer.append(theano.function(inputs=[x], outputs=layer_1))
        visual_layer.append(theano.shared(self.w1.get_value()))
        #pool 1
        visual_layer.append(theano.function(inputs=[x], outputs=layer_2))
        #conv 2
        visual_layer.append(theano.function(inputs=[x], outputs=layer_3))
        visual_layer.append(theano.shared(self.w2.get_value()))
        #pool 2
        visual_layer.append(theano.function(inputs=[x], outputs=layer_4))
        #conv 3
        visual_layer.append(theano.function(inputs=[x], outputs=layer_5))
        visual_layer.append(theano.shared(self.w3.get_value()))
        #pool 3
        visual_layer.append(theano.function(inputs=[x], outputs=layer_6))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_7))
        visual_layer.append(theano.function(inputs=[x], outputs=pyx))
        visual_layer.append(self.w_output)

        return  visual_layer

    def init_visual_model_3(self, x, layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, pyx):
        visual_layer = []
        visual_layer.append(theano.function(inputs=[x], outputs=layer_0))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_1))
        visual_layer.append(self.w1)
        visual_layer.append(theano.function(inputs=[x], outputs=layer_2))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_3))
        visual_layer.append(self.w2)
        visual_layer.append(theano.function(inputs=[x], outputs=layer_4))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_5))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_6))
        visual_layer.append(self.w3)
        visual_layer.append(theano.function(inputs=[x], outputs=pyx))
        visual_layer.append(self.w_output)

        return visual_layer

    def refresh_layout(self):
        print("asdasd")

    def remove_widget_layout(self):
        print(len(self.children()))
        count_component = len(self.children())
        for i in self.children():
            i.setParent(None)
        # for i in reversed(range(layout.count())):
        #     layout.itemAt(i).widget().setParent(None)

    def act_start(self):
        self.epoch = self.parent.ui_main.centralwidget.epochInput.getInt()
        print(self.epoch)
        training_thread = TrainingThread(self, self.epoch)
        training_thread.start()

    # def start_training(self):
    #
    #     print("Start Training")
    #     training_x, training_y, testing_x, testing_y = self.dataset.load_data_cifar10(batch=5)
    #     y_x = T.argmax(self.pyx, axis=1)
    #     training = Training(x=self.x, y=self.y, params=self.params, pyx=self.pyx, y_x=y_x, is_visual=False)
    #     training.training(training_x, training_y, testing_x, testing_y, self.vbox_visual)

# class TrainThread(QtCore.QThread):
#     def __init__(self):

