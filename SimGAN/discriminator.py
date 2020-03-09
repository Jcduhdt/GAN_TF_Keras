#!/usr/bin/env python3
import sys
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Lambda, Concatenate, MaxPooling2D
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, Nadam, Adamax
from keras.utils import plot_model
import tensorflow as tf
from SimGAN.loss import local_adversarial_loss


# simGAN的判别器是一个普通的卷积神经网络
# 在网络的最后部分做出了调整--输出模拟数据或真实数据的可能性
class Discriminator(object):
    def __init__(self, width=35, height=55, channels=1, name='discriminator'):
        self.W = width
        self.H = height
        self.C = channels
        self.SHAPE = (height, width, channels)
        self.NAME = name

        self.Discriminator = self.model()
        self.OPTIMIZER = SGD(lr=0.001)
        self.Discriminator.compile(loss=local_adversarial_loss, optimizer=self.OPTIMIZER)

        self.save_model_graph()
        self.summary()

    def model(self):
        # 定义模型并根据图片的尺寸创建一个输入层
        input_layer = Input(shape=self.SHAPE)
        # 网络以两个卷积层开始
        x = Conv2D(96, 3, strides=2, padding='same', activation='relu')(input_layer)
        x = Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
        # 使用作者推荐的一个池化层
        x = MaxPooling2D(pool_size=(3, 3), padding='same')(x)
        x = Conv2D(32, 3, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(32, 1, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(2, 1, strides=1, padding='same', activation='relu')(x)
        output_layer = Reshape((-1, 2))(x)
        return Model(input_layer, output_layer)

    def summary(self):
        return self.Discriminator.summary()

    def save_model_graph(self):
        plot_model(self.Discriminator, to_file='./out/Discriminator_Model.png')

    def save_model(self, epoch, batch):
        self.Discriminator.save('./out/' + self.NAME + '_Epoch_' + epoch + '_Batch_' + batch + 'model.h5')
