import theano
import theano.tensor as T
import numpy
import pickle as cPickle
import datetime
import time

from Plotter import Plotter


class Training():
    def __init__(self, x, y, y_x, pyx, params, is_visual):
        self.params = params
        self.cost = T.mean(T.nnet.categorical_crossentropy(pyx, y))
        self.update = self.RMSprop(cost=self.cost, params=self.params)
        self.train = theano.function(inputs=[x, y], outputs=self.cost, updates=self.update, allow_input_downcast=True)
        self.predict = theano.function(inputs=[x], outputs=y_x, allow_input_downcast=True)

        self.is_visual = is_visual

    """
    RMSProp (Tileman dan Hinton 2012)
    """
    def RMSprop(self, cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho * acc + (1 -  rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - lr * g))

        return updates

    def training(self, training_x, training_y, testing_x, testing_y, training_layout, dataset, p_epochs=100):
        print(training_layout.model)
        plotter = Plotter(dataset)
        # Ukuran file untuk 1 kali training
        batch_size = 200;
        epochs = p_epochs;
        label_train = numpy.argmax(testing_y, axis=1)
        print(p_epochs)

        for epoch in range(epochs):
            print("======== Epoch 1 ========")
            for start in range(0, len(training_x), batch_size):
                print("Batch Ke ", start)
                training_layout.progress_bar_value = training_layout.progress_bar_value + (100 / (epochs * 250))
                training_layout.progress_bar.setValue(training_layout.progress_bar_value)
                x_batch = training_x[start:start+batch_size]
                y_batch = training_y[start:start+batch_size]
                # plot.subplot(3, 8, 1); plot.imshow(datasets.to_RGB(x_batch[0], (32, 32, 3)))
                """
                Hitung cost
                """
                self.cost = self.train(x_batch, y_batch)
                print("Cost = ", self.cost)


                if(self.is_visual):
                    """
                    Visualize
                    """
                    #training_layout.input_label.setText(str(self.cost))
                    if(training_layout.model == 'Model 1'):
                        plotter.plot_training_model_1(x_batch[0:1], training_layout, self.params)
                    elif(training_layout.model == 'Model 2'):
                        plotter.plot_training_model_2(x_batch[0:1], training_layout, self.params)
                    else:
                        plotter.plot_training_model_3(x_batch[0:1], training_layout, self.params)
                    time.sleep(0.2)

            print("Prediction")
            prediction_test = self.predict(testing_x)
            print(prediction_test)
            accuracy = numpy.mean(prediction_test == label_train)
            print("Accuracy = ", accuracy)
            print("")
        now = datetime.datetime.now()
        date_now = ''+str(now.year)+str(now.month)+str(now.day)+'-'+str(now.hour)+str(now.minute)+str(now.second)
        f = open('../bobot/params-'+date_now, 'wb')
        cPickle.dump(self.params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
