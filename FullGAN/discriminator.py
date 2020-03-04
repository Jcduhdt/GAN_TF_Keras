#!/usr/bin/env python3
# GAN的判别器Discriminator

import sys
import numpy as np
# 导入所需的层
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
# 特殊的层结构LeakyReLU 特殊的激发层，用来确保当前单元在没有被激发的情况下，仍然存在一个较小的梯度。优于ReLU
from keras.layers.advanced_activations import LeakyReLU
# 导入顺序模型结构Sequential
from keras.models import Sequential, Model
# Adam优化器
from keras.optimizers import Adam
# 绘图，要安装pydot、graphviz.因为要绘制保存模型图
from keras.utils import plot_model


class Discriminator(object):
    # 使用width、height、channels和潜在空间尺寸(latent_size)来初始化类
    def __init__(self, width=28, height=28, channels=1, latent_size=100):
        # 加入输入参数作为类的中间变量
        self.CAPACITY = width * height * channels
        self.SHAPE = (width, height, channels)
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)

        # 根据后面定义的方法初始化模型
        self.Discriminator = self.model()
        # 对网络的学习过程进行配置
        # 使用binary_crossentropy作为损失函数来初始化模型，并指定优化器
        # 一般而言需要自己定义损失函数，这里就测试一下
        # metrics: 评价函数,与损失函数类似,只不过评价函数的结果不会用于训练过程中
        self.Discriminator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'])
        # 在终端展示模型的总结
        self.save_model()
        self.summary()

    # 判别器模型的定义
    def model(self):
        # 以一个顺序模型开始，可以以一种简单的方式来组合每一层
        # Keras会对这种方式的组合存在一些假设，如前一层的尺寸将会输入到下一层
        model = Sequential()
        # 第一层，将数据扁平化成一个单独的数据流 就是将图像转成一行
        model.add(Flatten(input_shape=self.SHAPE))
        # 该层为致密层，神经元的全链接层。神经网络中的一个基础模块，使得输入可以接触到每一个神经元
        model.add(Dense(self.CAPACITY, input_shape=self.SHAPE))
        model.add(LeakyReLU(alpha=0.2))
        # 该层容量减半，期望在当前层可以学习到一些重要特征
        model.add(Dense(int(self.CAPACITY / 2)))
        model.add(LeakyReLU(alpha=0.2))
        # 最后一层，表示输出是否属于某一类别的概率
        model.add(Dense(1, activation='sigmoid'))
        return model

    # Discriminator类中的辅助函数summary，save_model
    # model的summary方法将从Kersa中打印之间构建模型的总结信息
    def summary(self):
        return self.Discriminator.summary()

    # save_model keras工具类中plot_model的将以图像化的方式展示当前模型的结构，输入模型以及保存路径
    def save_model(self):
        plot_model(self.Discriminator.model, to_file='./data/Discriminator_Model.png')
