import numpy
import pandas
import theano

from Plotter import Plotter


class Validate():
    def __init__(self, x, pyx, y_x):

        self.test = theano.function(inputs=[x], outputs=pyx, allow_input_downcast=True)
        self.output = theano.function(inputs=[pyx], outputs=y_x, allow_input_downcast=True)

    def testing(self, dataset, testing_x, testing_y, widget):
        testing_x = testing_x
        label_testing = numpy.argmax(testing_y, axis=1)
        batch_size = 200
        plotter = Plotter(dataset)
        predict = numpy.zeros((10000, 10))
        predict_label = numpy.zeros(10000)
        completed = 0
        for start in range(0, len(testing_x), batch_size):
            print(start)
            completed = completed + 2;
            widget.progress_bar.setValue(completed)
            x_batch = testing_x[start:start+batch_size]
            predict[start:start+batch_size] = self.test(x_batch)

        predict_out = self.output(predict)

        pandas.DataFrame(predict).to_csv("../output_testing.csv")
        pandas.DataFrame(predict_out).to_csv("../output_testing_2.csv")

        accuracy = numpy.mean(predict_out == label_testing)
        print(accuracy)
        plotter.plot_testing(x_batch, predict[start:start+batch_size], widget.list_result)
        widget.acc_label.setText(str(accuracy))
        # print(predict.shape)

