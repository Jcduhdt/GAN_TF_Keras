#!/usr/bin/env python3
import sys
import numpy as np
from keras.layers import Dense, Reshape, Input, BatchNormalization
from keras.layers.core import Activation
from keras.layers.convolutional import UpSampling2D, Convolution2D, MaxPooling2D, Deconvolution2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, Nadam, Adamax
from keras import initializers
from keras.utils import plot_model


class Generator(object):
    def __init__(self, width=28, height=28, channels=1, latent_size=100, model_type='simple'):

        self.W = width
        self.H = height
        self.C = channels
        self.LATENT_SPACE_SIZE = latent_size
        self.latent_space = np.random.normal(0, 1, (self.LATENT_SPACE_SIZE,))

        # 根据参数得到需要的模型，一种是之前普通的模型，一种是卷积神经网络的模型
        if model_type == 'simple':
            self.Generator = self.model()
            self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)
            self.Generator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)
        elif model_type == 'DCGAN':
            self.Generator = self.dc_model()
            self.OPTIMIZER = Adam(lr=1e-4, beta_1=0.2)
            self.Generator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'])
        self.save_model()
        self.summary()

    def dc_model(self):

        model = Sequential()
        # 第一个致密层为使用LeakyReLu激发的输入层。输入尺寸为潜在空间的尺寸
        # 致密层的第一个参数为过滤器的初始化数量，可以调整这个参数来尝试不同网络。
        # 论文使用了1024个过滤器，为了更好地收敛，这里使用更多的过滤器 就是节点数吧
        model.add(Dense(12 * 12 * 768, input_dim=self.LATENT_SPACE_SIZE))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        # 对张量整型，以便于它可以用来表示一张图像
        # 过滤器的数量和整形层通道的数量的乘积是相同的
        # 每一层尺寸信息都必须和前面张量的尺寸信息相匹配
        # 输入8*8,对其上采样
        model.add(Reshape((12, 12, 768)))
        model.add(UpSampling2D())
        # 这一层输入16*16
        # 卷积核数目128、卷积核大小5*5
        # 将LeakyReLU作为激活函数传入，被提示错误，应当作为一层进行添加
        # model.add(Conv2D(128, 5, border_mode='same', activation=LeakyReLU(0.2)))
        model.add(Conv2D(384, 5, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(UpSampling2D())
        # 这一层输入32*32
        model.add(Conv2D(192, 5, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(UpSampling2D())
        # 上述几层包含以下特征
        # 1.跨中间层生成图像的2D卷积网络允许自定义过滤器大小、步长、边界模式，激活函数
        # 2.实践中LeakyReLu更好
        # 3.处理卷积层时使用BatchNormalization避免过高或过低激发，同时加速训练的过程并避免过拟合
        # 4.UpSampling2D默认进行扩展：16*16->32*32->64*64  有啥用啊关键是具体模型是啥样的

        # 还是要看上一层输出需要等于自己想要的图像大小
        # 输出图像
        model.add(Conv2D(self.C, 5, padding='same', activation='tanh'))

        return model

    def model(self, block_starting_size=128, num_blocks=4):
        model = Sequential()

        block_size = block_starting_size
        model.add(Dense(block_size, input_shape=(self.LATENT_SPACE_SIZE,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        for i in range(num_blocks - 1):
            block_size = block_size * 2
            model.add(Dense(block_size))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(self.W * self.H * self.C, activation='tanh'))
        model.add(Reshape((self.W, self.H, self.C)))

        return model

    def summary(self):
        return self.Generator.summary()

    def save_model(self):
        plot_model(self.Generator.model, to_file='./data/Generator_Model.png')
