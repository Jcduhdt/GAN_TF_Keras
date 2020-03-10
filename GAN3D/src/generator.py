#!/usr/bin/env python3
import sys
import numpy as np
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv3D, Deconv3D
from keras.layers import Input, BatchNormalization, Dense, Reshape
from keras.layers.core import Activation
from keras.optimizers import Adam, SGD
from keras.utils import plot_model


class Generator(object):
    def __init__(self, latent_size=100):
        # 以潜在空间的输入尺寸和SGD优化器初始化生成器类
        self.INPUT_SHAPE = (1, 1, 1, latent_size)
        # self.OPTIMIZER = Adam(lr=0.0001,beta_1=0.5)
        self.OPTIMIZER = SGD(lr=0.001, nesterov=True)

        self.Generator = self.model()
        self.Generator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)
        self.save_model()
        self.summary()

    # block方法用于模型的生成——主要用于创建模型架构中可复用的模板，包括Deconv3D层、BatchNorm以及ReLU激发
    def block(self, first_layer, filter_size=512, stride_size=(2, 2, 2), kernel_size=(4, 4, 4), padding='same'):
        x = Deconv3D(filters=filter_size, kernel_size=kernel_size,
                     strides=stride_size, kernel_initializer='glorot_normal',
                     bias_initializer='zeros', padding=padding)(first_layer)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)

        return x

    # 使用初始化阶段定义的尺寸来创建model方法
    def model(self):
        input_layer = Input(shape=self.INPUT_SHAPE)
        # 开始创建第一个block
        x = self.block(input_layer, filter_size=256, stride_size=(1, 1, 1), kernel_size=(4, 4, 4), padding='valid')
        # 创建第二个block，并将过滤器的数量减半
        x = self.block(x, filter_size=128, stride_size=(2, 2, 2), kernel_size=(4, 4, 4))
        # 最后一个block做一些调整，进行显式定义padding
        x = Deconv3D(filters=3, kernel_size=(4, 4, 4),
                     strides=(2, 2, 2), kernel_initializer='glorot_normal',
                     bias_initializer='zeros', padding='same')(x)
        x = BatchNormalization()(x)
        output_layer = Activation(activation='sigmoid')(x)
        # 创建模型
        model = Model(inputs=input_layer, outputs=output_layer)
        return model

    def summary(self):
        return self.Generator.summary()

    def save_model(self):
        plot_model(self.Generator, to_file='../data/Generator_Model.png')
