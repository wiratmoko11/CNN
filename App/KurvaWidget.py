from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class KurvaWidget(QWidget):
    def __init__(self, parent):
        super(KurvaWidget, self).__init__(parent)

    def _components(self):
        self.figureAccuracy = Figure()
        self.canvasAccuracy = FigureCanvas(figure=self.figureAccuracy)
        self.figureLoss = Figure()
        self.canvasLoss = FigureCanvas(figure=self.canvasLoss)

    def _layout(self):
        vbox_utama = QVBoxLayout()

        vbox_utama.addWidget(self.canvasAccuracy)
        vbox_utama.addWidget(self.canvasLoss)

        self.setLayout(vbox_utama)

    def plot_accuracy(self):
        """

        :return:
        """