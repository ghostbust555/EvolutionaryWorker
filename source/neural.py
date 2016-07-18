'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import numpy as np
from layertype import LayerType

np.random.seed(1337)  # for reproducibility


class Neural:


    def __init__(self, individual, thorough=False):
        self.batch_size = 128
        self.nb_classes = 10

        if thorough:
            self.nb_epoch = 12
        else:
            self.nb_epoch = 5


        # input image dimensions
        self.img_rows, self.img_cols = 28, 28
        # number of convolutional filters to use
        self.nb_filters = 32
        # size of pooling area for max pooling
        self.nb_pool = 2
        # convolution kernel size
        self.nb_conv = 3

        self.model = Sequential()
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()


        # the data, shuffled and split between train and test sets
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, self.img_rows, self.img_cols)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, self.img_rows, self.img_cols)
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255
        if not thorough:
            self.x_train = self.x_train[:500]
            self.x_test = self.x_test[500:750]
            self.y_train = self.y_train[:500]
            self.y_test = self.y_test[500:750]

        print('X_train shape:', self.x_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        self.y_train = np_utils.to_categorical(self.y_train, self.nb_classes)
        self.y_test = np_utils.to_categorical(self.y_test, self.nb_classes)

        self.dna2model(individual.dna)

        self.model.compile(loss=self.get_loss(individual.dna),
                           optimizer='adadelta',
                           metrics=['accuracy'])

    def get_loss(self, dna):
        for gene in dna:
            if gene[0] == LayerType.loss:
                return gene[1].name

        return 'categorical_crossentropy'

    def get_optimizer(self, dna):
        for gene in dna:
            if gene[0] == LayerType.optimizer:
                return gene[1]

        return 'adadelta'


    def run_network(self):
        self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, nb_epoch=self.nb_epoch,
                       verbose=1, validation_data=(self.x_test, self.y_test))

        score = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        return score[1]

    def dna2model(self, dna: list):
        self.model.add(Convolution2D(self.nb_filters, self.nb_conv, self.nb_conv,
                                     border_mode='valid',
                                     input_shape=(1, self.img_rows, self.img_cols)))

        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(self.nb_filters, self.nb_conv, self.nb_conv))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(self.nb_pool, self.nb_pool)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())

        print(dna)

        for gene in dna:
            x = gene[0]
            if x == LayerType.conv:
                pass
                # model.add(Convolution2D(self.nb_filters, self.nb_conv, self.nb_conv))
            elif x == LayerType.relu:
                self.model.add(Dense(gene[1]))
                self.model.add(Activation('relu'))
            elif x == LayerType.elu:
                self.model.add(Dense(gene[1]))
                self.model.add(ELU())
            elif x == LayerType.softmax:
                self.model.add(Dense(gene[1]))
                self.model.add(Activation('softmax'))
            elif x == LayerType.tanh:
                self.model.add(Dense(gene[1]))
                self.model.add(Activation('tanh'))
            elif x == LayerType.sigmoid:
                self.model.add(Dense(gene[1]))
                self.model.add(Activation('sigmoid'))
            elif x == LayerType.dropout:
                self.model.add(Dropout(min(gene[1]/2000, .8)))
            elif x == LayerType.maxpool:
                self.model.add(MaxPooling2D(pool_size=(self.nb_pool, self.nb_pool)))

        self.model.add(Dense(self.nb_classes))
        # model.add(Flatten())
