#!/usr/bin/env python3
import sys
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input
from keras.optimizers import Adam, SGD
from keras.utils import plot_model


class GAN(object):
    # lambda_cycle和lambda_id分别为X到Y的生成损失以及X到Y再到X的重构损失。
    # 根据论文提示，lambda_id应为lambda_cycle的10%
    def __init__(self, model_inputs=[], model_outputs=[], lambda_cycle=10.0, lambda_id=1.0):
        self.OPTIMIZER = SGD(lr=2e-4, nesterov=True)

        # self.inputs是由训练类初始化并传递给GAN的两个Keras输入数组
        self.inputs = model_inputs
        self.outputs = model_outputs
        self.gan_model = Model(self.inputs, self.outputs)
        self.OPTIMIZER = Adam(lr=2e-4, beta_1=0.5)
        # 输出数组是6个模型：其中四个生成器，两个判别器
        self.gan_model.compile(loss=['mse', 'mse',
                                     'mae', 'mae',
                                     'mae', 'mae'],
                               loss_weights=[1, 1,
                                             lambda_cycle, lambda_cycle,
                                             lambda_id, lambda_id],
                               optimizer=self.OPTIMIZER)
        self.save_model()
        self.summary()

    def model(self):
        model = Model()
        return model

    def summary(self):
        return self.gan_model.summary()

    def save_model(self):
        plot_model(self.gan_model, to_file='./data/GAN_Model.png')
