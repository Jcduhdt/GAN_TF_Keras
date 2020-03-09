#!/usr/bin/env python3
import tensorflow as tf


# simGAN作者推荐的自正则化损失函数
# 选择预测值和真实值之间标准化的绝对值
def self_regularization_loss(y_true, y_pred):
    return tf.multiply(0.0002, tf.reduce_sum(tf.abs(y_pred - y_true)))


# 局部对抗损失函数
# softmax_cross_entropy_with_logits_v2函数计算两个离散分类任务的错误概率，在这个例子中区分真是图像和模拟图像
def local_adversarial_loss(y_true, y_pred):
    truth = tf.reshape(y_true, (-1, 2))
    predicted = tf.reshape(y_pred, (-1, 2))

    computed_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=truth, logits=predicted)
    output = tf.reduce_mean(computed_loss)
    return output
