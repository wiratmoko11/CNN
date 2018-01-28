import matplotlib.pyplot as plot
class MyPlotter():
    def visual_kernel(self):
        ""
    def show_datasets(self, data):
        ""

    def visual_weight3(self, weigth_value):
        print(weigth_value.shape)
        vmin = weigth_value.min()
        vmax = weigth_value.max()

        kernel_numb = weigth_value.shape[0]
        kernel_column = weigth_value.shape[1]

        # plot.plot(4, 3)
        index = 1;

        for i in range(kernel_numb):
            for j in range(kernel_column):
                plot.subplot(kernel_numb, kernel_column, index); plot.imshow(weigth_value[i][j], vmin=vmin, vmax=vmax,  cmap=plot.cm.binary); plot.axis('off')
                index = index + 1


        plot.show()

    def visual_model1(self, visual_layer, plot_figure, canvas, x_batch, dataset):
        ax = plot_figure[0].add_subplot(111); ax.imshow(dataset.to_channel_last(x_batch[0], shape=(32,32,3)))


        ax_layer1 = plot_figure[1].add_subplot(141); ax_layer1.imshow(visual_layer[1](x_batch)[0][0], cmap=plot.cm.binary)
        ax_layer1 = plot_figure[1].add_subplot(142); ax_layer1.imshow(visual_layer[1](x_batch)[0][1], cmap=plot.cm.binary)
        ax_layer1 = plot_figure[1].add_subplot(143); ax_layer1.imshow(visual_layer[1](x_batch)[0][2], cmap=plot.cm.binary)
        ax_layer1 = plot_figure[1].add_subplot(144); ax_layer1.imshow(visual_layer[1](x_batch)[0][3], cmap=plot.cm.binary)

        ax_layer2 = plot_figure[2].add_subplot(141); ax_layer2.imshow(visual_layer[2](x_batch)[0][0], cmap=plot.cm.binary)
        ax_layer2 = plot_figure[2].add_subplot(142); ax_layer2.imshow(visual_layer[2](x_batch)[0][1], cmap=plot.cm.binary)
        ax_layer2 = plot_figure[2].add_subplot(143); ax_layer2.imshow(visual_layer[2](x_batch)[0][2], cmap=plot.cm.binary)
        ax_layer2 = plot_figure[2].add_subplot(144); ax_layer2.imshow(visual_layer[2](x_batch)[0][3], cmap=plot.cm.binary)

        ax_layer3 = plot_figure[3].add_subplot(181); ax_layer3.imshow(visual_layer[3](x_batch)[0][0], cmap=plot.cm.binary)
        ax_layer3 = plot_figure[3].add_subplot(182); ax_layer3.imshow(visual_layer[3](x_batch)[0][1], cmap=plot.cm.binary)
        ax_layer3 = plot_figure[3].add_subplot(183); ax_layer3.imshow(visual_layer[3](x_batch)[0][2], cmap=plot.cm.binary)
        ax_layer3 = plot_figure[3].add_subplot(184); ax_layer3.imshow(visual_layer[3](x_batch)[0][3], cmap=plot.cm.binary)
        ax_layer3 = plot_figure[3].add_subplot(185); ax_layer3.imshow(visual_layer[3](x_batch)[0][4], cmap=plot.cm.binary)
        ax_layer3 = plot_figure[3].add_subplot(186); ax_layer3.imshow(visual_layer[3](x_batch)[0][5], cmap=plot.cm.binary)
        ax_layer3 = plot_figure[3].add_subplot(187); ax_layer3.imshow(visual_layer[3](x_batch)[0][6], cmap=plot.cm.binary)
        ax_layer3 = plot_figure[3].add_subplot(188); ax_layer3.imshow(visual_layer[3](x_batch)[0][7], cmap=plot.cm.binary)

        ax_layer4 = plot_figure[4].add_subplot(181); ax_layer4.imshow(visual_layer[4](x_batch)[0][0], cmap=plot.cm.binary)
        ax_layer4 = plot_figure[4].add_subplot(182); ax_layer4.imshow(visual_layer[4](x_batch)[0][1], cmap=plot.cm.binary)
        ax_layer4 = plot_figure[4].add_subplot(183); ax_layer4.imshow(visual_layer[4](x_batch)[0][2], cmap=plot.cm.binary)
        ax_layer4 = plot_figure[4].add_subplot(184); ax_layer4.imshow(visual_layer[4](x_batch)[0][3], cmap=plot.cm.binary)
        ax_layer4 = plot_figure[4].add_subplot(185); ax_layer4.imshow(visual_layer[4](x_batch)[0][4], cmap=plot.cm.binary)
        ax_layer4 = plot_figure[4].add_subplot(186); ax_layer4.imshow(visual_layer[4](x_batch)[0][5], cmap=plot.cm.binary)
        ax_layer4 = plot_figure[4].add_subplot(187); ax_layer4.imshow(visual_layer[4](x_batch)[0][6], cmap=plot.cm.binary)
        ax_layer4 = plot_figure[4].add_subplot(188); ax_layer4.imshow(visual_layer[4](x_batch)[0][7], cmap=plot.cm.binary)

        print(visual_layer[1](x_batch).shape)
        print(visual_layer[2](x_batch).shape)
        print(visual_layer[3](x_batch).shape)
        print(visual_layer[4](x_batch).shape)
        print(visual_layer[5](x_batch).shape)
        print(visual_layer[6](x_batch).shape)
        print(visual_layer[7](x_batch)[0][9])
        ax_layer5 = plot_figure[5].add_subplot(1, 1, 1); ax_layer5.imshow(visual_layer[5](x_batch), cmap=plot.cm.binary)
        ax_layer6 = plot_figure[6].add_subplot(1, 1, 1); ax_layer6.imshow(visual_layer[6](x_batch), cmap=plot.cm.binary)
        ax_layer7 = plot_figure[7].add_subplot(1, 1, 1); ax_layer7.imshow(visual_layer[7](x_batch), cmap=plot.cm.binary)

        # for i in range(visual_layer[7](x_batch).shape[1]):
        #     np
        #     ax_layer7 = plot_figure[7].add_subplot(1, 10, (i+1)); ax_layer7.imshow(visual_layer[7](x_batch)[0][i], cmap=plot.cm.binary)



        canvas[0].draw()
        canvas[1].draw()
        canvas[2].draw()
        canvas[3].draw()
        canvas[4].draw()
        canvas[5].draw()
        canvas[6].draw()
        canvas[7].draw()

    def visual_layer(self, data_training, datasets):
        ""

        figure = plot.figure()
        ax = figure.add_subplot(8, 1, 1)
        ax_1 = ax.add_subplot(1, 1, 1)
        ax_1.imshow(datasets.to_channel_last(data_training[0], shape=(32, 32, 3)))
        ax2 = figure.add_subplot(8, 1, 2)
        ax2.imshow(datasets.to_channel_last(data_training[1], shape=(32, 32, 3)))
        # figure.add_axes(ax)


        # plot.subplot(1, 8, [1, 2]); plot.imshow(datasets.to_channel_last(data_training[0], shape=(32, 32, 3)))
        # plot.subplot(1, 8, [3, 4]); plot.imshow(datasets.to_channel_last(data_training[1], shape=(32, 32, 3)))
        # plot.subplot(1, 8, [5, 6]); plot.imshow(datasets.to_channel_last(data_training[2], shape=(32, 32, 3)))
        # plot.subplot(1, 8, [7, 8]); plot.imshow(datasets.to_channel_last(data_training[3], shape=(32, 32, 3)))

        plot.show()



