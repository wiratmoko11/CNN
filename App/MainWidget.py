from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QLineEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from App.TrainingWidget import TrainingWidget
from Datasets import Datasets


class MainWidget(QWidget):
    def __init__(self, parent, dataset, dataset_folder):
        super(MainWidget, self).__init__(parent)
        self.myApp = parent
        self.dataset_folder = dataset_folder
        self.dataset = dataset
        self.model = "Model 1"
        """

        """
        self.load_dataset()
        self._components()
        self._layout()

        self.plot_sample_image()

    def _components(self):
        """

        """
        self.datasetsLabel  = QLabel("Datasets")
        self.datasetsLabel.move(20, 50)
        self.datasetsInput = QLineEdit()
        self.datasetsInput.setText(self.dataset_folder)
        self.datasetsInput.setReadOnly(True)

        self.chooseModelLabel = QLabel("Pilih Model")
        self.chooseModelSelect = QComboBox()
        self.chooseModelSelect.addItem("Model 1")
        self.chooseModelSelect.addItem("Model 2")
        self.chooseModelSelect.addItem("Model 3")
        self.chooseModelSelect.activated.connect(self.handle_model_combobox)
        self.epochLabel  = QLabel("Jumlah Epoch")
        self.epochInput = QSpinBox()
        self.epochInput.setMaximum(1000)

        self.imageLeftButton = QPushButton("<<")
        self.imageLeftButton.clicked.connect(self.handle_backward)
        self.imageInput = QLineEdit()
        self.imageRightButton = QPushButton(">>")
        self.imageRightButton.clicked.connect(self.handle_forward)

        self.figureImageSample = Figure()
        self.canvasImageSample = FigureCanvas(figure=self.figureImageSample)

        self.buttonTrain = QPushButton("Training")

    def _layout(self):
        hbox_utama = QHBoxLayout()

        # vbox_form = QVBoxLayout()
        formbox_form = QFormLayout()



        formbox_form.addRow("Dataset", self.datasetsInput)
        formbox_form.addRow("Model", self.chooseModelSelect)
        formbox_form.addRow("Epoch", self.epochInput)
        formbox_form.addRow("", self.buttonTrain)

        vbox_data = QVBoxLayout()
        hbox_navigation_image = QHBoxLayout()
        hbox_navigation_image.addWidget(self.imageLeftButton)
        hbox_navigation_image.addWidget(self.imageInput)
        hbox_navigation_image.addWidget(self.imageRightButton)

        vbox_data.addLayout(hbox_navigation_image)
        vbox_data.addWidget(self.canvasImageSample)

        hbox_utama.addLayout(formbox_form)
        hbox_utama.addLayout(vbox_data)

        self.setLayout(hbox_utama)

    def load_dataset(self):
        self.training_x, self.training_y, self.testing_x, self.testing_y = self.dataset.load_data_cifar10(5)
        self.image_index = 0

    def plot_sample_image(self):
        self.plot = self.figureImageSample.add_subplot(111)
        self.plot.imshow(self.dataset.to_channel_last(self.training_x[self.image_index], shape=(32, 32, 3)))
        self.canvasImageSample.draw()

    def handle_forward(self):
        if(self.image_index < 50000):
            print('Forward')
            self.image_index = self.image_index + 1
            self.plot_sample_image()

    def handle_backward(self):
        if(self.image_index > 0):
            print('Backward')
            self.image_index = self.image_index - 1
            self.plot_sample_image()

    def handle_model_combobox(self, index):
        self.model = self.chooseModelSelect.itemText(index)
        self.myApp.ui_training.centralwidget.model = self.model
        self.myApp.ui_training.centralwidget.pilih_model()

