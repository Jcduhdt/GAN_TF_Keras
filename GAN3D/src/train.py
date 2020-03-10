#!/usr/bin/env python3
from GAN3D.src.gan import GAN
from GAN3D.src.generator import Generator
from GAN3D.src.discriminator import Discriminator
from keras.datasets import mnist
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import h5py
# This import registers the 3D projection, but is otherwise unused.
# 引入3D映射功能、没有引入其它不需要的功能
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


# from voxelgrid import VoxelGrid


class Trainer:
    # side:输入数据立方体的边长
    # latent:编码器输出尺寸
    # epochs:训练器在所有数据上迭代的次数
    # batch:在每个epoch中一次训练所需的数据
    # checkpoint:训练过程中绘制结果图片的频率
    def __init__(self, side=16, latent_size=32, epochs=100, batch=32, checkpoint=50, data_dir=''):
        self.SIDE = side
        self.EPOCHS = epochs
        self.BATCH = batch
        self.CHECKPOINT = checkpoint
        self.LATENT_SPACE_SIZE = latent_size
        self.LABELS = [1]
        # 使用内部方法载入类所需的数据
        self.load_3D_MNIST(data_dir)
        self.load_2D_encoded_MNIST()
        # 实例化generator和discriminator
        self.generator = Generator(latent_size=self.LATENT_SPACE_SIZE)
        self.discriminator = Discriminator(side=self.SIDE)
        # 实例化gan
        self.gan = GAN(generator=self.generator.Generator, discriminator=self.discriminator.Discriminator)

    # Translate data to color
    # 是translate的辅助函数————简化了从点阵到颜色映射的过程
    # 将数据映射成颜色
    def array_to_color(self, array, cmap="Oranges"):
        s_m = plt.cm.ScalarMappable(cmap=cmap)
        return s_m.to_rgba(array)[:, :-1]

    # 提供了将3D MNIST数据转换成三原色定义的x、y、z的功能
    def translate(self, x):
        xx = np.ndarray((x.shape[0], 4096, 3))
        for i in range(x.shape[0]):
            xx[i] = self.array_to_color(x[i])
        del x
        return xx

    # 以input_dir为参数加载h5格式的3D MNIST数据
    def load_3D_MNIST(self, input_dir):
        raw = h5py.File(input_dir, 'r')
        # 获取X_train数据，进行标准化并整形成模型所需要的格式
        self.X_train_3D = np.array(raw['X_train'])
        self.X_train_3D = (np.float32(self.X_train_3D) - 127.5) / 127.5
        self.X_train_3D = self.translate(self.X_train_3D).reshape(-1, 16, 16, 16, 3)
        # 同理对X_test
        self.X_test_3D = np.array(raw['X_test'])
        self.X_test_3D = (np.float32(self.X_test_3D) - 127.5) / 127.5
        self.X_test_3D = self.translate(self.X_test_3D).reshape(-1, 16, 16, 16, 3)
        # 需要额外保存Y数据集
        self.Y_train_3D = np.array(raw['y_train'])
        self.Y_test_3D = np.array(raw['y_test'])

        return

    # 载入前面使用编码器生成的npy文件
    def load_2D_encoded_MNIST(self):
        (_, self.Y_train_2D), (_, self.Y_test_2D) = mnist.load_data()
        self.X_train_2D_encoded = np.load('./x_train_encoded.npy')
        self.X_test_2D_encoded = np.load('./x_test_encoded.npy')
        return

    def train(self):
        # 伪造和真实对半开训练
        count_generated_images = int(self.BATCH / 2)
        count_real_images = int(self.BATCH / 2)
        for e in range(self.EPOCHS):
            for label in self.LABELS:
                # Grab the Real 3D Samples
                # 获取真实的3D样本
                all_3D_samples = self.X_train_3D[np.where(self.Y_train_3D == label)]
                # 随机获取一组数量小于batch大小的数据
                starting_index = randint(0, (len(all_3D_samples) - count_real_images))
                real_3D_samples = all_3D_samples[starting_index: int((starting_index + count_real_images))]
                # 通过一个全为1的数组来创建y_real_labels，对判别器来说是真
                y_real_labels = np.ones([count_generated_images, 1])

                # Grab Generated Images for this training batch
                # 同理获取对应数量的生成图片并生成一个全0数组
                all_encoded_samples = self.X_train_2D_encoded[np.where(self.Y_train_2D == label)]
                starting_index = randint(0, (len(all_encoded_samples) - count_generated_images))
                batch_encoded_samples = all_encoded_samples[
                                        starting_index: int((starting_index + count_generated_images))]
                # 对编码样本进行整形以适配模型的输入格式
                batch_encoded_samples = batch_encoded_samples.reshape(count_generated_images, 1, 1, 1,
                                                                      self.LATENT_SPACE_SIZE)

                x_generated_3D_samples = self.generator.Generator.predict(batch_encoded_samples)
                y_generated_labels = np.zeros([count_generated_images, 1])

                # Combine to train on the discriminator
                # 联合所有数据训练判别器
                x_batch = np.concatenate([real_3D_samples, x_generated_3D_samples])
                y_batch = np.concatenate([y_real_labels, y_generated_labels])

                # Now, train the discriminator with this batch
                # 训练当前batch的判别器
                self.discriminator.Discriminator.trainable = False
                discriminator_loss = self.discriminator.Discriminator.train_on_batch(x_batch, y_batch)[0]
                self.discriminator.Discriminator.trainable = True

                # Generate Noise
                # 使用随机索引选择编码样本并创建GAN训练数据
                starting_index = randint(0, (len(all_encoded_samples) - self.BATCH))
                x_batch_encoded_samples = all_encoded_samples[starting_index: int((starting_index + self.BATCH))]
                x_batch_encoded_samples = x_batch_encoded_samples.reshape(int(self.BATCH), 1, 1, 1,
                                                                          self.LATENT_SPACE_SIZE)
                y_generated_labels = np.ones([self.BATCH, 1])
                # 使用编码样本和生成标签来训练生成器
                generator_loss = self.gan.gan_model.train_on_batch(x_batch_encoded_samples, y_generated_labels)
                print('Epoch: ' + str(int(e)) + ' Label: ' + str(int(label)) + ', [Discriminator :: Loss: ' + str(
                    discriminator_loss) + '], [ Generator :: Loss: ' + str(generator_loss) + ']')
                if e % self.CHECKPOINT == 0 and e != 0:
                    self.plot_checkpoint(e, label)

        return

    def plot_checkpoint(self, e, label):
        filename = "../out/epoch_" + str(e) + "_label_" + str(label) + ".png"
        # 创建一个编码样本数组，对编码样本进行整形，以便当前的生成器使用
        all_encoded_samples = self.X_test_2D_encoded[np.where(self.Y_test_2D == label)]
        index = randint(0, (len(all_encoded_samples) - 1))
        batch_encoded_samples = all_encoded_samples[index]
        batch_encoded_samples = batch_encoded_samples.reshape(1, 1, 1, 1, self.LATENT_SPACE_SIZE)
        # 使用当前的生成器来生成带颜色的3D图像
        images = self.generator.Generator.predict(batch_encoded_samples)
        # 创建一个for循环迭代所有的像素，并保留颜色不是黑或白的像素
        xs = []
        ys = []
        zs = []
        cs = []
        for i in range(16):
            for j in range(16):
                for k in range(16):
                    color = images[0][i][j][k]
                    if np.mean(color) < 0.75 and np.mean(color) > 0.25:
                        xs.append(i)
                        ys.append(j)
                        zs.append(k)
                        cs.append(color)
        # 使用Matplotlib的scatter函数绘制保留下来的像素点————绘制完成后将文件保存在磁盘上
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(xs, ys, zs, alpha=0.1, c=cs)
        plt.savefig(filename)

        return
