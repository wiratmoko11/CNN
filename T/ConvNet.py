import numpy
import theano
import theano.tensor as T
import matplotlib.pyplot as plot
import time
import pickle as cPickle


from MyPlotter import MyPlotter


class ConvNet():
    """
        Covnet
        :parameter
            - x = data_training(batch, 3, width, height) tipe data T.tensor4
            y = label_training tipe data T.scalar
            y_x =
            pyx = output dari model
    """
    is_visual_model = True
    def __init__(self, x, y, y_x, pyx, params):

        self.plotter = MyPlotter()
        self.cost = T.mean(T.nnet.categorical_crossentropy(pyx, y))
        self.update = self.RMSprop(cost=self.cost, params=params)
        self.train = theano.function(inputs=[x, y], outputs=self.cost, updates=self.update, allow_input_downcast=True)

        self.testing = theano.function(inputs=[x, y], outputs=self.cost, allow_input_downcast=True)

        self.predict = theano.function(inputs=[x], outputs=y_x, allow_input_downcast=True)
        self.params = params
        # self.visual_layer = theano.function(inputs=[x], outputs=[])
    """
    Fungsi ubah nilai ke type float
    """
    def floatX(self, x):
        return numpy.asarray(x, dtype=theano.config.floatX)

    """
    RMSProp (Tileman dan Hinton 2012)
    """
    def RMSprop(self, cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - lr * g))

        return updates

    def training(self, training_x, training_y, testing_x, testing_y, visual_layer, plot_figure, canvas, dataset):
        # Ukuran file untuk 1 kali training
        batch_size = 200;
        epochs = 100;
        label_train = numpy.argmax(testing_y, axis=1)

        for epoch in range(epochs):
            print("======== Epoch 1 ========")
            for start in range(0, len(training_x), batch_size):
                print("Batch Ke ", start)
                x_batch = training_x[start:start+batch_size]
                y_batch = training_y[start:start+batch_size]
                # plot.subplot(3, 8, 1); plot.imshow(datasets.to_RGB(x_batch[0], (32, 32, 3)))
                """
                Hitung cost
                """
                self.cost = self.train(x_batch, y_batch)
                print("Cost = ", self.cost)

                if(self.is_visual_model):
                    # Visual Conv
                    self.plotter.visual_model1(visual_layer, plot_figure, canvas, x_batch, dataset)
                    time.sleep(0.5)

            print("Prediction")
            prediction_test = self.predict(testing_x)
            print(prediction_test)
            accuracy = numpy.mean(prediction_test == label_train)
            print("Accuracy = ", accuracy)
            print("")
        f = open('bobot/params', 'wb')
        cPickle.dump(self.params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


    def testing(self, testing_x, testing_y):
        """"

        """
        f = open('obj.save', 'rb')
        params = cPickle.load(f)
        f.close()
        label_testing = numpy.argmax(testing_y, axis=1)
        batch_size = 256
        for start in range(0, len(testing_x), batch_size):
            x_batch = testing_x[start:start+batch_size]
            y_batch = testing_y[start:start+batch_size]
            self.cost = self.testing(x_batch, y_batch)


        prediction = self.predict(testing_x)

        accuracy = numpy.mean(prediction == label_testing)
        print("Test accuracy = ", accuracy)





