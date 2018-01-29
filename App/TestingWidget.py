from PyQt5.QtWidgets import *
import pickle as cPickle
import theano.tensor as T
import numpy
from App.TestingDetailWindow import TestingDetailWindow
from ConvModel import ConvModel
from Utils import Utils
from Validate import Validate


class TestingWidget(QWidget):
    x = T.tensor4()
    y = T.matrix()
    def __init__(self, parent, dataset, dataset_folder):
        super(TestingWidget, self).__init__()
        self.utils = Utils()
        self.parent = parent
        self.dataset = dataset
        self.init_ui_components()
        self.init_ui_layout()
        self.model = "Model 1"

    def init_ui_components(self):
        self.load_model_label = QLabel("Load File Model")
        self.load_bobot_button = QPushButton("Load")
        self.load_bobot_button.clicked.connect(self.handle_load_model)
        self.load_model_path = QLabel("")
        self.acc_label = QLabel("0")


        self.chooseModelSelect = QComboBox()
        self.chooseModelSelect.addItem("Model 1")
        self.chooseModelSelect.addItem("Model 2")
        self.chooseModelSelect.addItem("Model 3")

        self.chooseModelSelect.activated.connect(self.handle_model_select)

        self.list_result = QListWidget()
        self.list_result.clicked.connect(self.handle_list_click);

        self.test_button = QPushButton("Testing")
        self.test_button.clicked.connect(self.handle_test)
        self.progress_bar = QProgressBar(self)


    def init_ui_layout(self):
        hbox_main = QHBoxLayout()

        vbox_form = QVBoxLayout()

        hbox_load_model = QHBoxLayout()
        # hbox_load_model.addWidget(self.load_model_label)
        # hbox_load_model.addWidget(self.load_model_button)
        # hbox_load_model.addWidget(self.load_model_path)
        form_form_layout = QFormLayout()
        form_form_layout.addRow("Pilih Model", self.chooseModelSelect)
        form_form_layout.addRow("Load File Bobot", self.load_bobot_button)
        form_form_layout.addRow("", self.test_button)

        vbox_form.addLayout(form_form_layout)
        # vbox_form.addWidget(self.test_button)

        vbox_result = QVBoxLayout()
        fl_acc = QFormLayout()
        fl_acc.addRow("Hasil Akurasi", self.acc_label)
        vbox_result.addLayout(fl_acc)
        vbox_result.addWidget(self.list_result)
        vbox_result.addWidget(self.progress_bar)

        hbox_main.addLayout(vbox_form)
        hbox_main.addLayout(vbox_result)

        self.setLayout(hbox_main)

    def handle_load_model(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Load File Model", "","All Files (*)", options=options)
        if fileName:
            print(fileName)
            self.filename = fileName
            # self.load_model(fileName)

    def handle_model_select(self, index):
        self.model = self.chooseModelSelect.itemText(index)
        print(self.model)

    def handle_test(self):
        conv_model = ConvModel()
        if(self.model == "Model 1"):
            self.load_bobot_1(self.filename)
            layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, self.pyx = conv_model.init_model_1(self.x, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output)
            self.start_testing()
        elif(self.model == "Model 2"):
            self.load_bobot_2(self.filename)
            layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7a, self.pyx = conv_model.init_model_2(self.x, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output)
            self.start_testing()
        elif(self.model == "Model 3"):
            self.load_bobot_3(self.filename)
            layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, self.pyx = conv_model.init_model_3(self.x, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_output, self.b_output)
            self.start_testing()

    def handle_list_click(self, item):
        print("CLick")
        index_data = self.list_result.row(self.list_result.currentItem()) + 9800
        training_x, training_y, testing_x, testing_y = self.dataset.load_data_cifar10(5)
        self.citra = numpy.zeros((1, 3, 32, 32))
        self.citra[0] = testing_x[index_data]
        self.true_label = testing_y[index_data]
        self.parent.start_ui_testing_detail()
        # ui_detail_testing = TestingDetailWindow()
        # ui_detail_testing.setup_ui(citra, true_label, self.dataset)

        # ui_detail_testing.show()

    def start_testing(self):
        training_x, training_y, testing_x, testing_y = self.dataset.load_data_cifar10(5)
        y_x = T.argmax(self.pyx, axis=1)
        validate  = Validate(x=self.x, pyx=self.pyx, y_x=y_x)
        validate.testing(dataset=self.dataset, testing_x=testing_x, testing_y=testing_y, widget=self)

    def load_bobot_1(self, file):
        f = open(file, 'rb')
        self.params = cPickle.load(f)
        print(self.params)
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
        print(len(self.params))
        self.w1 = self.params[0]
        self.b1 = self.utils.init_kernel((16,))
        self.w2 = self.params[1]
        self.b2 = self.utils.init_kernel((20,))
        self.w3 = self.params[2]
        self.b3 = self.utils.init_kernel((20,))
        self.w_output = self.params[3]
        self.b_output = self.utils.init_kernel((10,))

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
    # def init_list(self):
