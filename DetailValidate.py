import theano

from Plotter import Plotter


class DetailValidate():
    def __init__(self, x, pyx, y_x):
        self.test = theano.function(inputs=[x], outputs=pyx, allow_input_downcast=True)
        self.output = theano.function(inputs=[pyx], outputs=y_x, allow_input_downcast=True)

    def validate(self, citra, true_label, widget):
        plotter = Plotter(widget.dataset)
        if(widget.model == 'Model 1'):
            plotter.plot_detail_testing_1(citra, true_label, widget)
        elif(widget.model == 'Model 2'):
            plotter.plot_detail_testing_2(citra, true_label, widget)
        elif(widget.model == 'Model 3'):
            plotter.plot_detail_testing_3(citra, true_label, widget)

        predict = self.test(citra)
        predict_out = self.output(predict)

        print(predict_out)
