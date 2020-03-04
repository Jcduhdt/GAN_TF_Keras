#!/usr/bin/env python3
import sys
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Lambda, Concatenate
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, Nadam, Adamax
import keras.backend as K
from keras.utils import plot_model


class Discriminator(object):
    def __init__(self, width=28, height=28, channels=1, starting_filters=64):
        self.W = width
        self.H = height
        self.C = channels
        self.CAPACITY = width * height * channels
        self.SHAPE = (width, height, channels)
        self.FS = starting_filters  # FilterStart

        self.Discriminator = self.model()
        self.OPTIMIZER = Adam(lr=2e-4, beta_1=0.5, decay=1e-5)
        self.Discriminator.compile(loss='mse', optimizer=self.OPTIMIZER, metrics=['accuracy'])

        self.save_model()
        self.summary()

    def model(self):
        # 该模型在每一层增加过滤器的数量，有什么规律可以确定这个吗
        # 模型将两张图片作为输入，并将其连接输入到一个张量中
        input_A = Input(shape=self.SHAPE)
        input_B = Input(shape=self.SHAPE)
        input_layer = Concatenate(axis=-1)([input_A, input_B])
        # 256*512
        up_layer_1 = Conv2D(self.FS, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(
            input_layer)
        # 128*256
        up_layer_2 = Conv2D(self.FS * 2, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(
            up_layer_1)
        leaky_layer_2 = BatchNormalization(momentum=0.8)(up_layer_2)
        # 64*128
        up_layer_3 = Conv2D(self.FS * 4, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(
            leaky_layer_2)
        leaky_layer_3 = BatchNormalization(momentum=0.8)(up_layer_3)
        # 32*64
        up_layer_4 = Conv2D(self.FS * 8, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(
            leaky_layer_3)
        leaky_layer_4 = BatchNormalization(momentum=0.8)(up_layer_4)
        # 16*32
        # 最终输出层是一个01或者真伪的二元分类器，因该size要与输入对应确定层数吧，才能得到一个输出的结果
        output_layer = Conv2D(1, kernel_size=4, strides=1, padding='same')(leaky_layer_4)
        # emmm，这输出是我计算错了吗。
        return Model([input_A, input_B], output_layer)

    def summary(self):
        return self.Discriminator.summary()

    def save_model(self):
        plot_model(self.Discriminator, to_file='./out/Discriminator_Model.png')
