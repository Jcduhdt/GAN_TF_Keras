#!/usr/bin/env python3
# 整个GAN模型

import sys
import numpy as np
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import plot_model


class GAN(object):
    # 初始化导入discriminator、generator
    def __init__(self, discriminator, generator):
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)

        self.Generator = generator

        self.Discriminator = discriminator
        # 判别器可被训练性置成False,意味着在对抗训练的过程中，判别器不会被训练
        # 生成器会随着训练逐渐变强，而判别器则保持不变。在这个架构中，这一步对模型收敛十分重要
        self.Discriminator.trainable = False

        self.gan_model = self.model()
        # 对网络的学习过程进行配置
        self.gan_model.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)
        self.save_model()
        self.summary()

    def model(self):
        model = Sequential()
        model.add(self.Generator)
        model.add(self.Discriminator)
        return model

    def summary(self):
        return self.gan_model.summary()

    def save_model(self):
        plot_model(self.gan_model.model, to_file='./data/GAN_Model.png')
