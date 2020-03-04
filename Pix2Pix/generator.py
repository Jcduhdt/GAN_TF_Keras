#!/usr/bin/env python
import sys
import numpy as np
from keras.layers import Dense, Reshape, Input, BatchNormalization, Concatenate
from keras.layers.core import Activation
from keras.layers.convolutional import UpSampling2D, Convolution2D, MaxPooling2D, Deconvolution2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, Nadam, Adamax
from keras import initializers
from keras.utils import plot_model


class Generator(object):
    def __init__(self, width=28, height=28, channels=1):
        self.W = width
        self.H = height
        self.C = channels
        self.SHAPE = (width, height, channels)

        self.Generator = self.model()
        self.OPTIMIZER = Adam(lr=2e-4, beta_1=0.5, decay=1e-5)
        self.Generator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'])

        self.save_model()
        self.summary()

    def model(self):
        input_layer = Input(shape=self.SHAPE)

        # 前三层实现编码器的前几层
        # strides就相当于两倍下采样，编码器只是过滤器数目的变化，但经过每层，图片的像素都会下采样2
        # 最后的()应该是表示上一层
        down_1 = Conv2D(64, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(input_layer)

        down_2 = Conv2D(64 * 2, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(down_1)
        norm_2 = BatchNormalization()(down_2)

        down_3 = Conv2D(64 * 4, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(norm_2)
        norm_3 = BatchNormalization()(down_3)

        # 在解码器之前加入相同的四层
        down_4 = Conv2D(64 * 8, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(norm_3)
        norm_4 = BatchNormalization()(down_4)

        down_5 = Conv2D(64 * 8, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(norm_4)
        norm_5 = BatchNormalization()(down_5)

        down_6 = Conv2D(64 * 8, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(norm_5)
        norm_6 = BatchNormalization()(down_6)

        down_7 = Conv2D(64 * 8, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(norm_6)
        norm_7 = BatchNormalization()(down_7)

        # 使用相同大小的过滤器，并开始对编码器的输出进行上采样
        # 注意：在Concatenate层忽略了一些连接
        # concatenate从名字可以看出是连接两个数组，这里是连接两层，应该是大小必须要对应，所以根据上采样和下采样的对应关系连接对应层
        upsample_1 = UpSampling2D(size=2)(norm_7)
        up_conv_1 = Conv2D(64 * 8, kernel_size=4, strides=1, padding='same', activation='relu')(upsample_1)
        norm_up_1 = BatchNormalization(momentum=0.8)(up_conv_1)
        add_skip_1 = Concatenate()([norm_up_1, norm_6])

        upsample_2 = UpSampling2D(size=2)(add_skip_1)
        up_conv_2 = Conv2D(64 * 8, kernel_size=4, strides=1, padding='same', activation='relu')(upsample_2)
        norm_up_2 = BatchNormalization(momentum=0.8)(up_conv_2)
        add_skip_2 = Concatenate()([norm_up_2, norm_5])

        upsample_3 = UpSampling2D(size=2)(add_skip_2)
        up_conv_3 = Conv2D(64 * 8, kernel_size=4, strides=1, padding='same', activation='relu')(upsample_3)
        norm_up_3 = BatchNormalization(momentum=0.8)(up_conv_3)
        add_skip_3 = Concatenate()([norm_up_3, norm_4])

        upsample_4 = UpSampling2D(size=2)(add_skip_3)
        up_conv_4 = Conv2D(64 * 4, kernel_size=4, strides=1, padding='same', activation='relu')(upsample_4)
        norm_up_4 = BatchNormalization(momentum=0.8)(up_conv_4)
        add_skip_4 = Concatenate()([norm_up_4, norm_3])

        upsample_5 = UpSampling2D(size=2)(add_skip_4)
        up_conv_5 = Conv2D(64 * 2, kernel_size=4, strides=1, padding='same', activation='relu')(upsample_5)
        norm_up_5 = BatchNormalization(momentum=0.8)(up_conv_5)
        add_skip_5 = Concatenate()([norm_up_5, norm_2])

        upsample_6 = UpSampling2D(size=2)(add_skip_5)
        up_conv_6 = Conv2D(64, kernel_size=4, strides=1, padding='same', activation='relu')(upsample_6)
        norm_up_6 = BatchNormalization(momentum=0.8)(up_conv_6)
        add_skip_6 = Concatenate()([norm_up_6, down_1])

        last_upsample = UpSampling2D(size=2)(add_skip_6)
        output_layer = Conv2D(self.C, kernel_size=4, strides=1, padding='same', activation='tanh')(last_upsample)

        return Model(input_layer, output_layer)

    def summary(self):
        return self.Generator.summary()

    def save_model(self):
        plot_model(self.Generator, to_file='./out/Generator_Model.png')
