#!/usr/bin/env python3
# GAN的生成器generator

import sys
# 用于生成噪点
import numpy as np
from keras.layers import Dense, Reshape
# BatchNormalization通过标准化前一层的激发来对层进行清理，可提升整个网络的效率
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import plot_model


class Generator(object):
    # 初始化输入变量width、height、channels。latent_size也是一个帮助定义分布尺寸的重要参数，神经网络之后会利用该参数进行采样
    def __init__(self, width=28, height=28, channels=1, latent_size=100):
        self.W = width
        self.H = height
        self.C = channels
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)

        self.LATENT_SPACE_SIZE = latent_size
        self.latent_space = np.random.normal(0, 1, (self.LATENT_SPACE_SIZE,))

        self.Generator = self.model()
        self.Generator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)
        self.save_model()
        self.summary()

    def model(self, block_starting_size=128, num_blocks=4):
        # 使用基本的Sequential结构定义模型
        model = Sequential()
        # 网络第一层，加入和潜在空间尺寸相同的致密层，并用这个尺寸作为初始块的尺寸
        block_size = block_starting_size
        model.add(Dense(block_size, input_shape=(self.LATENT_SPACE_SIZE,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        # 添加后续的层，均为致密层，每次层的大小翻倍，可对这里改进，使用不同大小的block
        for i in range(num_blocks - 1):
            block_size = block_size * 2
            model.add(Dense(block_size))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))

        # 最后一层将输出层构造成以生成和输入数据相同尺寸的结果，并返回模型
        model.add(Dense(self.W * self.H * self.C, activation='tanh'))
        model.add(Reshape((self.W, self.H, self.C)))

        return model

    def summary(self):
        return self.Generator.summary()

    def save_model(self):
        plot_model(self.Generator.model, to_file='./data/Generator_Model.png')
