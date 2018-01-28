from PyQt5.QtWidgets import *
import theano
import theano.tensor as T
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pickle as cPickle

from ConvModel import ConvModel
from DetailValidate import DetailValidate


class TestingDetailWidget(QWidget):
    x = T.tensor4()
    y = T.matrix()

    def __init__(self, parent, citra, true_label, dataset, file_bobot, model):
        super(TestingDetailWidget, self).__init__(parent)
        print("asdasdasd");
        self.dataset = dataset
        self.init_ui_components()
        # self.init_ui_layout()
        self.file_bobot = file_bobot
        self.model = model
        self.pilih_model()
        y_x = T.argmax(self.pyx, axis=1)
        detail_validation = DetailValidate(self.x, self.pyx, y_x)
        detail_validation.validate(citra, true_label, self)

    def pilih_model(self):
        conv_model = ConvModel()
        if(self.model == "Model 1"):
            self._component_model_1()
            self._layout_model_1()
            self.load_bobot_1(self.file_bobot)
            layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, self.pyx = conv_model.init_model_1(self.x, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output)
            self.visual_layer = self.init_visual_model_1(self.x, layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, self.pyx)
        elif(self.model == "Model 2"):
            self._component_model_2()
            self._layout_model_2()
            self.load_bobot_2(self.file_bobot)
            layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7a, self.pyx = conv_model.init_model_2(self.x, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output)
            self.visual_layer = self.init_visual_model_2(self.x, layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7a, self.pyx)
        else:
            self._component_model_3()
            self._layout_model_3()
            self.load_bobot_3(self.file_bobot)
            layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, self.pyx = conv_model.init_model_3(self.x, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output)
            self.visual_layer = self.init_visual_model_3(self.x, layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, self.pyx)


    def init_ui_components(self):
        """"""
        self.list_result = QListWidget()

    def _component_model_1(self):
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

    def _component_model_2(self):
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
        """"""
        vbox_main = QVBoxLayout()
        vbox_main.addWidget(self.list_result)

        self.setLayout(vbox_main)

    def _layout_model_2(self):
        """"""
        vbox_main = QVBoxLayout()
        vbox_main.addWidget(self.list_result)

        self.setLayout(vbox_main)

    def _layout_model_3(self):
        """"""
        vbox_main = QVBoxLayout()
        vbox_main.addWidget(self.list_result)

        self.setLayout(vbox_main)

    def init_visual_model_1(self, x, layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, pyx):
        visual_layer = []
        visual_layer.append(theano.function(inputs=[x], outputs=layer_0))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_1, allow_input_downcast=True))
        visual_layer.append(theano.shared(self.w1.get_value()))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_2, allow_input_downcast=True))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_3, allow_input_downcast=True))
        visual_layer.append(self.w2)
        visual_layer.append(theano.function(inputs=[x], outputs=layer_4, allow_input_downcast=True))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_5, allow_input_downcast=True))
        visual_layer.append(theano.function(inputs=[x], outputs=layer_6, allow_input_downcast=True))
        visual_layer.append(self.w3)
        visual_layer.append(theano.function(inputs=[x], outputs=pyx, allow_input_downcast=True))
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


    def init_ui_layout(self):
        """

        :return:
        """

    def load_bobot_1(self, file):
        f = open(file, 'rb')
        self.params = cPickle.load(f)
        f.close()
        self.w1 = self.params[0]
        self.b1 = self.params[1]
        self.w2 = self.params[2]
        self.b2 = self.params[3]
        self.w3 = self.params[4]
        self.b3 = self.params[5]
        self.w_output = self.params[6]
        self.b_output = self.params[7]

    def load_bobot_2(self, file):
        f = open(file, 'rb')
        self.params = cPickle.load(f)
        f.close()
        self.w1 = self.params[0]
        # self.b1 = self.params[1]
        self.w2 = self.params[1]
        # self.b2 = self.params[3]
        self.w3 = self.params[2]
        # self.b3 = self.params[5]
        self.w_output = self.params[3]
        # self.b_output = self.params[7]

    def load_bobot_3(self, file):
        f = open(file, 'rb')
        self.params = cPickle.load(f)
        f.close()
        self.w1 = self.params[0]
        self.b1 = self.params[1]
        self.w2 = self.params[2]
        self.b2 = self.params[3]
        self.w3 = self.params[4]
        self.b3 = self.params[5]
        self.w_output = self.params[6]
        self.b_output = self.params[7]
