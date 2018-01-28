from PyQt5.QtWidgets import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class ResultWidget(QWidget):
    def __init__(self):
        super(ResultWidget, self).__init__()

        self.init_ui()
        self.init_layout()

    def init_ui(self):

        self.label_number = QLabel("")

        self.figure_image = Figure()
        self.canvas_image = FigureCanvas(figure=self.figure_image)

        self.label_prediksi_1 = QLabel("")
        self.label_prediksi_2 = QLabel("")
        self.label_prediksi_3 = QLabel("")

        self.label_prediksi_kategori = QLabel()

    def init_layout(self):
        vbox_main = QVBoxLayout()

        vbox_result = QVBoxLayout()
        hbox_items = QHBoxLayout()
        hbox_items.addWidget(self.label_number)
        hbox_items.addWidget(self.canvas_image)
        hbox_items.setSpacing(0)
        vbox_prediksi = QVBoxLayout()
        vbox_prediksi.addWidget(self.label_prediksi_1)
        vbox_prediksi.addWidget(self.label_prediksi_2)
        vbox_prediksi.addWidget(self.label_prediksi_3)

        hbox_items.addLayout(vbox_prediksi)

        vbox_result.addLayout(hbox_items)
        vbox_main.addLayout(vbox_result)
        self.setLayout(vbox_main)