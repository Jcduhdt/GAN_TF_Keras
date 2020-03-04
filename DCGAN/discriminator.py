#!/usr/bin/env python3
import sys
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Lambda, concatenate
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, Nadam, Adamax
import keras.backend as K
from keras.utils import plot_model


class Discriminator(object):
    def __init__(self, width=28, height=28, channels=1, latent_size=100, model_type='simple'):
        self.W = width
        self.H = height
        self.C = channels
        self.CAPACITY = width * height * channels
        self.SHAPE = (width, height, channels)

        if model_type == 'simple':
            self.Discriminator = self.model()
            self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)
            self.Discriminator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'])
        elif model_type == 'DCGAN':
            self.Discriminator = self.dc_model()
            self.OPTIMIZER = Adam(lr=1e-4, beta_1=0.2)
            self.Discriminator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'])

        self.save_model()
        self.summary()

    def dc_model(self):
        model = Sequential()
        # 64个5*5的过滤器，2*2的向下采样以及一个输入图像的尺寸
        model.add(Conv2D(96, 5, strides=(2, 2), input_shape=(self.W, self.H, self.C), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # 每次迭代强行稀疏，随即丢弃30%的学习到的圈中，保证不过拟合，并学习到关键特征
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

        model.add(Conv2D(192, 5, strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

        model.add(Conv2D(384, 5, strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        # 扁平化
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model

    def model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.SHAPE))
        model.add(Dense(self.CAPACITY, input_shape=self.SHAPE))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(int(self.CAPACITY / 2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def summary(self):
        return self.Discriminator.summary()

    def save_model(self):
        plot_model(self.Discriminator.model, to_file='./data/Discriminator_Model.png')
