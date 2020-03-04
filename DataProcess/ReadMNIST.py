# -*- coding:utf-8 -*-
# @Time : 2020/2/28 17:46
# @Author : Zhang
# @File : ReadMNIST.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 读取目录下的数据集
mnist = input_data.read_data_sets("E:/tensorflow/data_set/MNIST_data/", one_hot=False)
# 查看训练数据中的图片的形状
print("Shape of the Image Training Data is " + str(mnist.train.images.shape))
# 查看训练数据中的标签的形状
print("Shape of the Label Training Data is " + str(mnist.train.labels.shape))
# 从数据集中随机选取一个样本
index = np.random.choice(mnist.train.images.shape[0], 1)
random_image = mnist.train.images[index]
random_label = mnist.train.labels[index]
random_image = random_image.reshape([28, 28])

# 绘制图片
plt.gray()
plt.imshow(random_image)
plt.show()
