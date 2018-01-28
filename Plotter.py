import matplotlib.pyplot as plot
import numpy
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import *

from App.ResultWidget import ResultWidget



class Plotter():
    def __init__(self, dataset):

        self.dataset = dataset
        self.CIFAR10_LABELS_LIST = [
            'airplane',
            'automobile',
            'bird',
            'cat',
            'deer',
            'dog',
            'frog',
            'horse',
            'ship',
            'truck'
        ]

    def plot_training_model_1(self, x_batch, training_layout):

        ax = training_layout.figure_input.add_subplot(111); ax.imshow(self.dataset.to_channel_last(x_batch[0], shape=(32,32,3)))

        ax_layer_conv1 = training_layout.figure_conv1.add_subplot(141); ax_layer_conv1.imshow(training_layout.visual_layer[1](x_batch)[0][0], cmap=plot.cm.binary)
        ax_layer_conv1 = training_layout.figure_conv1.add_subplot(142); ax_layer_conv1.imshow(training_layout.visual_layer[1](x_batch)[0][1], cmap=plot.cm.binary)
        ax_layer_conv1 = training_layout.figure_conv1.add_subplot(143); ax_layer_conv1.imshow(training_layout.visual_layer[1](x_batch)[0][2], cmap=plot.cm.binary)
        ax_layer_conv1 = training_layout.figure_conv1.add_subplot(144); ax_layer_conv1.imshow(training_layout.visual_layer[1](x_batch)[0][3], cmap=plot.cm.binary)

        count_w1 = 1
        for i in range(training_layout.visual_layer[2].eval().shape[0]):
            for j in range(training_layout.visual_layer[2].eval().shape[1]):
                ax_w1 = training_layout.figure_w1.add_subplot(training_layout.visual_layer[2].eval().shape[0], training_layout.visual_layer[2].eval().shape[1], count_w1); ax_w1.imshow(training_layout.visual_layer[2].eval()[i][j], cmap=plot.cm.binary)
                ax_w1.set_axis_off()
                count_w1 = count_w1 + 1


        ax_layer_pool1 = training_layout.figure_pooling1.add_subplot(141); ax_layer_pool1.imshow(training_layout.visual_layer[3](x_batch)[0][0], cmap=plot.cm.binary)
        ax_layer_pool1 = training_layout.figure_pooling1.add_subplot(142); ax_layer_pool1.imshow(training_layout.visual_layer[3](x_batch)[0][1], cmap=plot.cm.binary)
        ax_layer_pool1 = training_layout.figure_pooling1.add_subplot(143); ax_layer_pool1.imshow(training_layout.visual_layer[3](x_batch)[0][2], cmap=plot.cm.binary)
        ax_layer_pool1 = training_layout.figure_pooling1.add_subplot(144); ax_layer_pool1.imshow(training_layout.visual_layer[3](x_batch)[0][3], cmap=plot.cm.binary)

        count_w2 = 1
        for i in range(training_layout.visual_layer[5].eval().shape[0]):
            for j in range(training_layout.visual_layer[5].eval().shape[1]):
                ax_w2 = training_layout.figure_w2.add_subplot(training_layout.visual_layer[5].eval().shape[0], training_layout.visual_layer[5].eval().shape[1], count_w2); ax_w2.imshow(training_layout.visual_layer[5].eval()[i][j], cmap=plot.cm.binary)
                ax_w2.set_axis_off()
                count_w2 = count_w2 + 1

        ax_layer_conv2 = training_layout.figure_conv2.add_subplot(181); ax_layer_conv2.imshow(training_layout.visual_layer[4](x_batch)[0][0], cmap=plot.cm.binary)
        ax_layer_conv2 = training_layout.figure_conv2.add_subplot(182); ax_layer_conv2.imshow(training_layout.visual_layer[4](x_batch)[0][1], cmap=plot.cm.binary)
        ax_layer_conv2 = training_layout.figure_conv2.add_subplot(183); ax_layer_conv2.imshow(training_layout.visual_layer[4](x_batch)[0][2], cmap=plot.cm.binary)
        ax_layer_conv2 = training_layout.figure_conv2.add_subplot(184); ax_layer_conv2.imshow(training_layout.visual_layer[4](x_batch)[0][3], cmap=plot.cm.binary)
        ax_layer_conv2 = training_layout.figure_conv2.add_subplot(185); ax_layer_conv2.imshow(training_layout.visual_layer[4](x_batch)[0][4], cmap=plot.cm.binary)
        ax_layer_conv2 = training_layout.figure_conv2.add_subplot(186); ax_layer_conv2.imshow(training_layout.visual_layer[4](x_batch)[0][5], cmap=plot.cm.binary)
        ax_layer_conv2 = training_layout.figure_conv2.add_subplot(187); ax_layer_conv2.imshow(training_layout.visual_layer[4](x_batch)[0][6], cmap=plot.cm.binary)
        ax_layer_conv2 = training_layout.figure_conv2.add_subplot(188); ax_layer_conv2.imshow(training_layout.visual_layer[4](x_batch)[0][7], cmap=plot.cm.binary)

        ax_layer_pool2 = training_layout.figure_pooling2.add_subplot(181); ax_layer_pool2.imshow(training_layout.visual_layer[6](x_batch)[0][0], cmap=plot.cm.binary)
        ax_layer_pool2 = training_layout.figure_pooling2.add_subplot(182); ax_layer_pool2.imshow(training_layout.visual_layer[6](x_batch)[0][1], cmap=plot.cm.binary)
        ax_layer_pool2 = training_layout.figure_pooling2.add_subplot(183); ax_layer_pool2.imshow(training_layout.visual_layer[6](x_batch)[0][2], cmap=plot.cm.binary)
        ax_layer_pool2 = training_layout.figure_pooling2.add_subplot(184); ax_layer_pool2.imshow(training_layout.visual_layer[6](x_batch)[0][3], cmap=plot.cm.binary)
        ax_layer_pool2 = training_layout.figure_pooling2.add_subplot(185); ax_layer_pool2.imshow(training_layout.visual_layer[6](x_batch)[0][4], cmap=plot.cm.binary)
        ax_layer_pool2 = training_layout.figure_pooling2.add_subplot(186); ax_layer_pool2.imshow(training_layout.visual_layer[6](x_batch)[0][5], cmap=plot.cm.binary)
        ax_layer_pool2 = training_layout.figure_pooling2.add_subplot(187); ax_layer_pool2.imshow(training_layout.visual_layer[6](x_batch)[0][6], cmap=plot.cm.binary)
        ax_layer_pool2 = training_layout.figure_pooling2.add_subplot(188); ax_layer_pool2.imshow(training_layout.visual_layer[6](x_batch)[0][7], cmap=plot.cm.binary)

        fc1 = numpy.zeros((1, 200))
        fc1[0] = training_layout.visual_layer[7](x_batch)[0]

        ax_fc1 = training_layout.figure_fc1.add_subplot(111); ax_fc1.imshow(fc1, cmap=plot.cm.binary)

        fc2 = numpy.zeros((1, 100))
        fc2[0] = training_layout.visual_layer[8](x_batch)[0]

        ax_fc2 = training_layout.figure_fc2.add_subplot(111); ax_fc2.imshow(fc2, cmap=plot.cm.binary)

        softmax = numpy.zeros((1, 10))
        softmax[0] = training_layout.visual_layer[10](x_batch)[0]
        print(softmax.shape)
        ax_layer_softmax = training_layout.figure_softmax.add_subplot(111); ax_layer_softmax.imshow(softmax, cmap=plot.cm.binary)



        print(numpy.argmax(softmax, axis=1)[0])
        print(print(self.CIFAR10_LABELS_LIST.__getitem__(numpy.argmax(softmax, axis=1)[0])))

        training_layout.canvas_input.draw()
        training_layout.canvas_conv1.draw()
        training_layout.canvas_w1.draw()
        training_layout.canvas_pooling1.draw()
        training_layout.canvas_w2.draw()
        training_layout.canvas_conv2.draw()
        training_layout.canvas_pooling2.draw()
        training_layout.canvas_fc1.draw()
        training_layout.canvas_fc2.draw()
        training_layout.canvas_softmax.draw()


    def plot_training_model_2(self, x_batch, training_layout):
        print("Plot 2")
        """

        :param x_batch:
        :param training_layout:
        :return:
        """
        #input
        ax = training_layout.figure_input.add_subplot(111); ax.imshow(self.dataset.to_channel_last(x_batch[0], shape=(32,32,3)))

        #W1
        count_w1 = 1
        for i in range(training_layout.visual_layer[2].eval().shape[0]):
            for j in range(training_layout.visual_layer[2].eval().shape[1]):
                ax_w1 = training_layout.figure_w1.add_subplot(training_layout.visual_layer[2].eval().shape[0], training_layout.visual_layer[2].eval().shape[1], count_w1); ax_w1.imshow(training_layout.visual_layer[2].eval()[i][j], cmap=plot.cm.binary)
                ax_w1.set_axis_off()
                count_w1 = count_w1 + 1
        # Conv1
        for i in range(16):
            ax_layer_conv1 = training_layout.figure_conv1.add_subplot(1, 16, (i+1)); ax_layer_conv1.imshow(training_layout.visual_layer[1](x_batch)[0][i], cmap=plot.cm.binary)
            ax_layer_conv1.set_axis_off()

        #Pooling 1
        for i in range(16):
            ax_layer_pool1 = training_layout.figure_pooling1.add_subplot(1, 16, (i+1)); ax_layer_pool1.imshow(training_layout.visual_layer[3](x_batch)[0][i], cmap=plot.cm.binary)
            ax_layer_pool1.set_axis_off()

        #W2
        count_w2 = 1
        for i in range(training_layout.visual_layer[2].eval().shape[0]):
            for j in range(training_layout.visual_layer[2].eval().shape[1]):
                ax_w2 = training_layout.figure_w2.add_subplot(training_layout.visual_layer[5].eval().shape[0], training_layout.visual_layer[5].eval().shape[1], count_w2); ax_w2.imshow(training_layout.visual_layer[5].eval()[i][j], cmap=plot.cm.binary)
                ax_w2.set_axis_off()
                count_w2 = count_w2 + 1

        #Conv2
        for i in range(20):
            ax_layer_conv2 = training_layout.figure_conv2.add_subplot(1, 20, (i+1)); ax_layer_conv2.imshow(training_layout.visual_layer[4](x_batch)[0][i], cmap=plot.cm.binary)
            ax_layer_conv2.set_axis_off()

        #Pooling 2
        for i in range(20):
            ax_layer_pool2 = training_layout.figure_pooling2.add_subplot(1, 20, (i+1)); ax_layer_pool2.imshow(training_layout.visual_layer[6](x_batch)[0][i], cmap=plot.cm.binary)
            ax_layer_pool2.set_axis_off()

        #W3
        count_w3 = 1
        for i in range(training_layout.visual_layer[2].eval().shape[0]):
            for j in range(training_layout.visual_layer[2].eval().shape[1]):
                ax_w3 = training_layout.figure_w3.add_subplot(training_layout.visual_layer[8].eval().shape[0], training_layout.visual_layer[8].eval().shape[1], count_w3); ax_w2.imshow(training_layout.visual_layer[8].eval()[i][j], cmap=plot.cm.binary)
                ax_w3.set_axis_off()
                count_w3 = count_w3 + 1

        #Conv3
        for i in range(20):
            ax_layer_conv3 = training_layout.figure_conv3.add_subplot(1, 20, (i+1)); ax_layer_conv3.imshow(training_layout.visual_layer[7](x_batch)[0][i], cmap=plot.cm.binary)
            ax_layer_conv3.set_axis_off()

        #Pooling 3
        for i in range(20):
            ax_layer_pool3 = training_layout.figure_pooling3.add_subplot(1, 20, (i+1)); ax_layer_pool3.imshow(training_layout.visual_layer[9](x_batch)[0][i], cmap=plot.cm.binary)
            ax_layer_pool3.set_axis_off()

        #fc 1
        fc1 = numpy.zeros((1, 320))
        fc1[0] = training_layout.visual_layer[10](x_batch)[0]
        ax_fc1 = training_layout.figure_fc1.add_subplot(111); ax_fc1.imshow(fc1, cmap=plot.cm.binary)
        ax_fc1.set_axis_off()

        softmax = numpy.zeros((1, 10))
        softmax[0] = training_layout.visual_layer[11](x_batch)[0]
        print(softmax.shape)
        ax_layer_softmax = training_layout.figure_softmax.add_subplot(111); ax_layer_softmax.imshow(softmax, cmap=plot.cm.binary)

        training_layout.canvas_conv1.draw()
        training_layout.canvas_pooling1.draw()
        training_layout.canvas_input.draw()
        training_layout.canvas_w1.draw()
        training_layout.canvas_w2.draw()
        training_layout.canvas_conv2.draw()
        training_layout.canvas_pooling2.draw()
        training_layout.canvas_w3.draw()
        training_layout.canvas_conv3.draw()
        training_layout.canvas_pooling3.draw()
        training_layout.canvas_fc1.draw()
        training_layout.canvas_softmax.draw()

    def plot_training_model_3(self, x_batch, training_layout):
        ax_input = training_layout.figure_input.add_subplot(111); ax_input.imshow(self.dataset.to_channel_last(x_batch[0], shape=(32, 32, 3)))
        ax_input.set_axis_off()

        training_layout.canvas_input.draw()

        #W1
        count_w1 = 1
        for i in range(training_layout.visual_layer[2].eval().shape[0]):
            for j in range(training_layout.visual_layer[2].eval().shape[1]):
                ax_w1 = training_layout.figure_w1.add_subplot(training_layout.visual_layer[2].eval().shape[0], training_layout.visual_layer[2].eval().shape[1], count_w1); ax_w1.imshow(training_layout.visual_layer[2].eval()[i][j], cmap=plot.cm.binary)
                ax_w1.set_axis_off()
                count_w1 = count_w1 + 1

        # Conv1
        for i in range(16):
            ax_layer_conv1 = training_layout.figure_conv1.add_subplot(1, 16, (i+1)); ax_layer_conv1.imshow(training_layout.visual_layer[1](x_batch)[0][i], cmap=plot.cm.binary)
            ax_layer_conv1.set_axis_off()

        #Pooling 1
        for i in range(16):
            ax_layer_pool1 = training_layout.figure_pooling1.add_subplot(1, 16, (i+1)); ax_layer_pool1.imshow(training_layout.visual_layer[3](x_batch)[0][i], cmap=plot.cm.binary)
            ax_layer_pool1.set_axis_off()

        #W2
        count_w2 = 1
        for i in range(training_layout.visual_layer[5].eval().shape[0]):
            for j in range(training_layout.visual_layer[5].eval().shape[1]):
                ax_w2 = training_layout.figure_w2.add_subplot(training_layout.visual_layer[5].eval().shape[0], training_layout.visual_layer[5].eval().shape[1], count_w2); ax_w2.imshow(training_layout.visual_layer[5].eval()[i][j], cmap=plot.cm.binary)
                ax_w2.set_axis_off()
                count_w2 = count_w2 + 1

        #Conv2
        for i in range(20):
            ax_layer_conv2 = training_layout.figure_conv2.add_subplot(1, 20, (i+1)); ax_layer_conv2.imshow(training_layout.visual_layer[4](x_batch)[0][i], cmap=plot.cm.binary)
            ax_layer_conv2.set_axis_off()


        #Pooling 2
        for i in range(20):
            ax_layer_pool2 = training_layout.figure_pooling2.add_subplot(1, 20, (i+1)); ax_layer_pool2.imshow(training_layout.visual_layer[6](x_batch)[0][i], cmap=plot.cm.binary)
            ax_layer_pool2.set_axis_off()


        fc1 = numpy.zeros((1, 1280))
        fc1[0] = training_layout.visual_layer[7](x_batch)[0]
        ax_fc1 = training_layout.figure_fc1.add_subplot(111); ax_fc1.imshow(fc1, cmap=plot.cm.binary)
        ax_fc1.set_axis_off()

        fc2 = numpy.zeros((1, 128))
        fc2[0] = training_layout.visual_layer[8](x_batch)[0]
        ax_fc2 = training_layout.figure_fc2.add_subplot(111); ax_fc2.imshow(fc2, cmap=plot.cm.binary)
        ax_fc2.set_axis_off()

        softmax = numpy.zeros((1, 10))
        softmax[0] = training_layout.visual_layer[10](x_batch)[0]
        print(softmax.shape)
        ax_layer_softmax = training_layout.figure_softmax.add_subplot(111); ax_layer_softmax.imshow(softmax, cmap=plot.cm.binary)


    def plot_testing(self, x_batch, predict, list_widget):
        list_widget.clear()
        """

        :param x_batch:
        :param predict:
        :return:
        """
        for i in range(len(x_batch)):
            top_3_predict = numpy.argsort(predict[i])[::-1][:3]
            #print(predict)
            result_widget = ResultWidget()
            result_widget.label_number.setText(str(i+1))
            category_1 = self.CIFAR10_LABELS_LIST.__getitem__(top_3_predict[0])
            value_1 = predict[i][top_3_predict[0]]
            category_2 = self.CIFAR10_LABELS_LIST.__getitem__(top_3_predict[1])
            value_2 = predict[i][top_3_predict[1]]
            category_3 = self.CIFAR10_LABELS_LIST.__getitem__(top_3_predict[2])
            value_3 = predict[i][top_3_predict[2]]
            result_widget.label_prediksi_1.setText(category_1+" - "+str(value_1)+"")
            result_widget.label_prediksi_2.setText(category_2+" - "+str(value_2)+"")
            result_widget.label_prediksi_3.setText(category_3+" - "+str(value_3)+"")

            # result_widget.canvas_image.
            ax = result_widget.figure_image.add_subplot(111); ax.imshow(self.dataset.to_channel_last(x_batch[i], shape=(32,32,3)))
            ax.set_position([0.1, 0.2, 0.8, 0.8])
            ax.set_axis_off()
            result_widget.canvas_image.draw()
            list_item_widget = QListWidgetItem(list_widget)
            # print(result_widget.sizeHint())
            list_size = QSize()
            list_size.setHeight(100)
            list_size.setWidth(502)
            list_item_widget.setSizeHint(list_size)

            list_widget.addItem(list_item_widget)
            list_widget.setItemWidget(list_item_widget, result_widget)

    def plot_detail_testing_1(self, citra, true_label, detail_testing_widget):
        """"""

        ax_input = detail_testing_widget.figure_input.add_subplot(111); ax_input.imshow(self.dataset.to_channel_last(citra[0], shape=(32, 32, 3)))
        ax_input.set_axis_off()

        detail_testing_widget.canvas_input.draw()

        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_size = QSize()
        list_size.setHeight(100)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_input)

        #W1
        count_w1 = 1
        for i in range(detail_testing_widget.visual_layer[2].eval().shape[0]):
            for j in range(detail_testing_widget.visual_layer[2].eval().shape[1]):
                ax_w1 = detail_testing_widget.figure_w1.add_subplot(detail_testing_widget.visual_layer[2].eval().shape[0], detail_testing_widget.visual_layer[2].eval().shape[1], count_w1); ax_w1.imshow(detail_testing_widget.visual_layer[2].eval()[i][j], cmap=plot.cm.binary)
                ax_w1.set_axis_off()
                count_w1 = count_w1 + 1

        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_w1)

        # Conv1
        print(citra.shape)
        for i in range(4):
            ax_layer_conv1 = detail_testing_widget.figure_conv1.add_subplot(1, 4, (i+1)); ax_layer_conv1.imshow(detail_testing_widget.visual_layer[1](citra)[0][i], cmap=plot.cm.binary)
            ax_layer_conv1.set_axis_off()

        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_conv1)

        #Pooling 1
        for i in range(4):
            ax_layer_pool1 = detail_testing_widget.figure_pooling1.add_subplot(1, 4, (i+1)); ax_layer_pool1.imshow(detail_testing_widget.visual_layer[3](citra)[0][i], cmap=plot.cm.binary)
            ax_layer_pool1.set_axis_off()

        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_pooling1)

        #W2
        count_w2 = 1
        for i in range(detail_testing_widget.visual_layer[5].eval().shape[0]):
            for j in range(detail_testing_widget.visual_layer[5].eval().shape[1]):
                ax_w2 = detail_testing_widget.figure_w2.add_subplot(detail_testing_widget.visual_layer[5].eval().shape[0], detail_testing_widget.visual_layer[5].eval().shape[1], count_w2); ax_w2.imshow(detail_testing_widget.visual_layer[5].eval()[i][j], cmap=plot.cm.binary)
                ax_w2.set_axis_off()
                count_w2 = count_w2 + 1

        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_w2)

        #Conv2
        for i in range(8):
            ax_layer_conv2 = detail_testing_widget.figure_conv2.add_subplot(1, 8, (i+1)); ax_layer_conv2.imshow(detail_testing_widget.visual_layer[4](citra)[0][i], cmap=plot.cm.binary)
            ax_layer_conv2.set_axis_off()

        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_conv2)

        #Pooling 2
        for i in range(8):
            ax_layer_pool2 = detail_testing_widget.figure_pooling2.add_subplot(1, 8, (i+1)); ax_layer_pool2.imshow(detail_testing_widget.visual_layer[6](citra)[0][i], cmap=plot.cm.binary)
            ax_layer_pool2.set_axis_off()

        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_pooling2)

        fc1 = numpy.zeros((1, 200))
        fc1[0] = detail_testing_widget.visual_layer[7](citra)[0]
        ax_fc1 = detail_testing_widget.figure_fc1.add_subplot(111); ax_fc1.imshow(fc1, cmap=plot.cm.binary)
        ax_fc1.set_axis_off()
        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_fc1)



        fc2 = numpy.zeros((1, 100))
        fc2[0] = detail_testing_widget.visual_layer[8](citra)[0]
        ax_fc2 = detail_testing_widget.figure_fc2.add_subplot(111); ax_fc2.imshow(fc2, cmap=plot.cm.binary)
        ax_fc2.set_axis_off()
        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_fc2)

        softmax = numpy.zeros((1, 10))
        softmax[0] = detail_testing_widget.visual_layer[10](citra)[0]
        print(softmax.shape)
        ax_layer_softmax = detail_testing_widget.figure_softmax.add_subplot(111); ax_layer_softmax.imshow(softmax, cmap=plot.cm.binary)
        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_softmax)

    def plot_detail_testing_2(self, citra, true_label, detail_testing_widget):

        ax_input = detail_testing_widget.figure_input.add_subplot(111); ax_input.imshow(self.dataset.to_channel_last(citra[0], shape=(32, 32, 3)))
        ax_input.set_axis_off()

        detail_testing_widget.canvas_input.draw()

        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_size = QSize()
        list_size.setHeight(100)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_input)

        count_w1 = 1
        for i in range(detail_testing_widget.visual_layer[2].eval().shape[0]):
            for j in range(detail_testing_widget.visual_layer[2].eval().shape[1]):
                ax_w1 = detail_testing_widget.figure_w1.add_subplot(detail_testing_widget.visual_layer[2].eval().shape[0], detail_testing_widget.visual_layer[2].eval().shape[1], count_w1); ax_w1.imshow(detail_testing_widget.visual_layer[2].eval()[i][j], cmap=plot.cm.binary)
                ax_w1.set_axis_off()
                count_w1 = count_w1 + 1
        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_w1)

        # Conv1
        for i in range(16):
            ax_layer_conv1 = detail_testing_widget.figure_conv1.add_subplot(1, 16, (i+1)); ax_layer_conv1.imshow(detail_testing_widget.visual_layer[1](citra)[0][i], cmap=plot.cm.binary)
            ax_layer_conv1.set_axis_off()
        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_conv1)

        #Pooling 1
        for i in range(16):
            ax_layer_pool1 = detail_testing_widget.figure_pooling1.add_subplot(1, 16, (i+1)); ax_layer_pool1.imshow(detail_testing_widget.visual_layer[3](citra)[0][i], cmap=plot.cm.binary)
            ax_layer_pool1.set_axis_off()
        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_pooling1)
        #W2
        count_w2 = 1
        for i in range(detail_testing_widget.visual_layer[2].eval().shape[0]):
            for j in range(detail_testing_widget.visual_layer[2].eval().shape[1]):
                ax_w2 = detail_testing_widget.figure_w2.add_subplot(detail_testing_widget.visual_layer[5].eval().shape[0], detail_testing_widget.visual_layer[5].eval().shape[1], count_w2); ax_w2.imshow(detail_testing_widget.visual_layer[5].eval()[i][j], cmap=plot.cm.binary)
                ax_w2.set_axis_off()
                count_w2 = count_w2 + 1
        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_w2)


        #Conv2
        for i in range(20):
            ax_layer_conv2 = detail_testing_widget.figure_conv2.add_subplot(1, 20, (i+1)); ax_layer_conv2.imshow(detail_testing_widget.visual_layer[4](citra)[0][i], cmap=plot.cm.binary)
            ax_layer_conv2.set_axis_off()
        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_conv2)


        #Pooling 2
        for i in range(20):
            ax_layer_pool2 = detail_testing_widget.figure_pooling2.add_subplot(1, 20, (i+1)); ax_layer_pool2.imshow(detail_testing_widget.visual_layer[6](citra)[0][i], cmap=plot.cm.binary)
            ax_layer_pool2.set_axis_off()
        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_pooling2)


        #W3
        count_w3 = 1
        for i in range(detail_testing_widget.visual_layer[2].eval().shape[0]):
            for j in range(detail_testing_widget.visual_layer[2].eval().shape[1]):
                ax_w3 = detail_testing_widget.figure_w3.add_subplot(detail_testing_widget.visual_layer[8].eval().shape[0], detail_testing_widget.visual_layer[8].eval().shape[1], count_w3); ax_w2.imshow(detail_testing_widget.visual_layer[8].eval()[i][j], cmap=plot.cm.binary)
                ax_w3.set_axis_off()
                count_w3 = count_w3 + 1
        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_w3)


        #Conv3
        for i in range(20):
            ax_layer_conv3 = detail_testing_widget.figure_conv3.add_subplot(1, 20, (i+1)); ax_layer_conv3.imshow(detail_testing_widget.visual_layer[7](citra)[0][i], cmap=plot.cm.binary)
            ax_layer_conv3.set_axis_off()
        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_conv3)


        #Pooling 3
        for i in range(20):
            ax_layer_pool3 = detail_testing_widget.figure_pooling3.add_subplot(1, 20, (i+1)); ax_layer_pool3.imshow(detail_testing_widget.visual_layer[9](citra)[0][i], cmap=plot.cm.binary)
            ax_layer_pool3.set_axis_off()
        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_pooling3)


        #fc 1
        fc1 = numpy.zeros((1, 320))
        fc1[0] = detail_testing_widget.visual_layer[10](citra)[0]
        ax_fc1 = detail_testing_widget.figure_fc1.add_subplot(111); ax_fc1.imshow(fc1, cmap=plot.cm.binary)
        ax_fc1.set_axis_off()
        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_fc1)


        softmax = numpy.zeros((1, 10))
        softmax[0] = detail_testing_widget.visual_layer[11](citra)[0]
        print(softmax.shape)
        ax_layer_softmax = detail_testing_widget.figure_softmax.add_subplot(111); ax_layer_softmax.imshow(softmax, cmap=plot.cm.binary)
        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_softmax)


    def plot_detail_testing_3(self, citra, true_label, detail_testing_widget):
        ax_input = detail_testing_widget.figure_input.add_subplot(111); ax_input.imshow(self.dataset.to_channel_last(citra[0], shape=(32, 32, 3)))
        ax_input.set_axis_off()

        detail_testing_widget.canvas_input.draw()

        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_size = QSize()
        list_size.setHeight(100)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_input)

        #W1
        count_w1 = 1
        for i in range(detail_testing_widget.visual_layer[2].eval().shape[0]):
            for j in range(detail_testing_widget.visual_layer[2].eval().shape[1]):
                ax_w1 = detail_testing_widget.figure_w1.add_subplot(detail_testing_widget.visual_layer[2].eval().shape[0], detail_testing_widget.visual_layer[2].eval().shape[1], count_w1); ax_w1.imshow(detail_testing_widget.visual_layer[2].eval()[i][j], cmap=plot.cm.binary)
                ax_w1.set_axis_off()
                count_w1 = count_w1 + 1

        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_w1)

        # Conv1
        print(citra.shape)
        for i in range(16):
            ax_layer_conv1 = detail_testing_widget.figure_conv1.add_subplot(1, 16, (i+1)); ax_layer_conv1.imshow(detail_testing_widget.visual_layer[1](citra)[0][i], cmap=plot.cm.binary)
            ax_layer_conv1.set_axis_off()

        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_conv1)

        #Pooling 1
        for i in range(16):
            ax_layer_pool1 = detail_testing_widget.figure_pooling1.add_subplot(1, 16, (i+1)); ax_layer_pool1.imshow(detail_testing_widget.visual_layer[3](citra)[0][i], cmap=plot.cm.binary)
            ax_layer_pool1.set_axis_off()

        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_pooling1)

        #W2
        count_w2 = 1
        for i in range(detail_testing_widget.visual_layer[5].eval().shape[0]):
            for j in range(detail_testing_widget.visual_layer[5].eval().shape[1]):
                ax_w2 = detail_testing_widget.figure_w2.add_subplot(detail_testing_widget.visual_layer[5].eval().shape[0], detail_testing_widget.visual_layer[5].eval().shape[1], count_w2); ax_w2.imshow(detail_testing_widget.visual_layer[5].eval()[i][j], cmap=plot.cm.binary)
                ax_w2.set_axis_off()
                count_w2 = count_w2 + 1

        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_w2)

        #Conv2
        for i in range(20):
            ax_layer_conv2 = detail_testing_widget.figure_conv2.add_subplot(1, 20, (i+1)); ax_layer_conv2.imshow(detail_testing_widget.visual_layer[4](citra)[0][i], cmap=plot.cm.binary)
            ax_layer_conv2.set_axis_off()

        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_conv2)

        #Pooling 2
        for i in range(20):
            ax_layer_pool2 = detail_testing_widget.figure_pooling2.add_subplot(1, 20, (i+1)); ax_layer_pool2.imshow(detail_testing_widget.visual_layer[6](citra)[0][i], cmap=plot.cm.binary)
            ax_layer_pool2.set_axis_off()

        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_pooling2)

        fc1 = numpy.zeros((1, 1280))
        fc1[0] = detail_testing_widget.visual_layer[7](citra)[0]
        ax_fc1 = detail_testing_widget.figure_fc1.add_subplot(111); ax_fc1.imshow(fc1, cmap=plot.cm.binary)
        ax_fc1.set_axis_off()
        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_fc1)



        fc2 = numpy.zeros((1, 128))
        fc2[0] = detail_testing_widget.visual_layer[8](citra)[0]
        ax_fc2 = detail_testing_widget.figure_fc2.add_subplot(111); ax_fc2.imshow(fc2, cmap=plot.cm.binary)
        ax_fc2.set_axis_off()
        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_fc2)

        softmax = numpy.zeros((1, 10))
        softmax[0] = detail_testing_widget.visual_layer[10](citra)[0]
        print(softmax.shape)
        ax_layer_softmax = detail_testing_widget.figure_softmax.add_subplot(111); ax_layer_softmax.imshow(softmax, cmap=plot.cm.binary)
        list_item_widget = QListWidgetItem(detail_testing_widget.list_result)
        list_item_widget.setSizeHint(list_size)
        detail_testing_widget.list_result.addItem(list_item_widget)
        detail_testing_widget.list_result.setItemWidget(list_item_widget, detail_testing_widget.canvas_softmax)
