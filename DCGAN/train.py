#!/usr/bin/env python3
from DCGAN.gan import GAN
from DCGAN.generator import Generator
from DCGAN.discriminator import Discriminator
from keras.datasets import mnist
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class Trainer:
    def __init__(self, width=28, height=28, channels=1, latent_size=100, epochs=50000, batch=32, checkpoint=50,
                 model_type=-1, data_path=''):
        self.W = width
        self.H = height
        self.C = channels
        self.EPOCHS = epochs
        self.BATCH = batch
        self.CHECKPOINT = checkpoint
        self.model_type = model_type

        self.LATENT_SPACE_SIZE = latent_size

        self.generator = Generator(height=self.H, width=self.W, channels=self.C, latent_size=self.LATENT_SPACE_SIZE,
                                   model_type='DCGAN')
        self.discriminator = Discriminator(height=self.H, width=self.W, channels=self.C, model_type='DCGAN')
        self.gan = GAN(generator=self.generator.Generator, discriminator=self.discriminator.Discriminator)

        # self.load_MNIST()
        self.load_npy(data_path)

    # 注意
    # 1.数据已经经过清理和整形--意味着样本的结构、通道数、高度、宽度都已经调整成学习可以直接使用的格式
    # 2.图片的尺寸都相同
    # 3.所能使用的数据量依赖于机器的性能；
        # 例如，一个1060GPU、16GB RAM的机器一次可以加载大约50%的数据。这需要根据自己的系统性能来调整amount_of_data参数
    def load_npy(self, npy_path):
        self.X_train = np.load(npy_path)
        self.X_train = self.X_train[:int(0.25 * float(len(self.X_train)))]
        self.X_train = (self.X_train.astype(np.float32) - 127.5) / 127.5
        self.X_train = np.expand_dims(self.X_train, axis=3)
        return

    def load_MNIST(self, model_type=3):
        allowed_types = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if self.model_type not in allowed_types:
            print('ERROR: Only Integer Values from -1 to 9 are allowed')

        (self.X_train, self.Y_train), (_, _) = mnist.load_data()
        if self.model_type != -1:
            self.X_train = self.X_train[np.where(self.Y_train == int(self.model_type))[0]]

        # Rescale -1 to 1
        # Find Normalize Function from CV Class  
        self.X_train = (np.float32(self.X_train) - 127.5) / 127.5
        self.X_train = np.expand_dims(self.X_train, axis=3)
        return

    def train(self):
        for e in range(self.EPOCHS):
            # 变量b 用来记录完成一个epoch所需要的batch数
            b = 0
            discriminator_loss = 0
            generator_loss = 0
            X_train_temp = deepcopy(self.X_train)
            while len(X_train_temp) > self.BATCH:
                # Keep track of Batches
                # 不断更新变量记录epoch中使用的batch数
                b = b + 1

                # Train Discriminator
                # Make the training batch for this model be half real, half noise
                # Grab Real Images for this training batch
                # 实现一个可以分别在真实数据(if中的true)和生成数据(false)中训练的训练器
                # 训练判别器
                if self.flipCoin():
                    # 训练中随机采样，并从复制的数组中删除这个样本
                    count_real_images = int(self.BATCH)
                    starting_index = randint(0, (len(X_train_temp) - count_real_images))
                    real_images_raw = X_train_temp[starting_index: (starting_index + count_real_images)]
                    # self.plot_check_batch(b,real_images_raw) # 这将会排除大量图片 不懂什么意思
                    # Delete the images used until we have none left
                    # 不断删除图片，直到数组为空 为啥啊
                    X_train_temp = np.delete(X_train_temp, range(starting_index, (starting_index + count_real_images)), 0)
                    x_batch = real_images_raw.reshape(count_real_images, self.W, self.H, self.C)
                    y_batch = np.ones([count_real_images, 1])
                else:
                    # Grab Generated Images for this training batch
                    latent_space_samples = self.sample_latent_space(self.BATCH)
                    x_batch = self.generator.Generator.predict(latent_space_samples)
                    # 打标签，假0
                    y_batch = np.zeros([self.BATCH, 1])

                # Now, train the discriminator with this batch
                # 使用这个batch的数据训练discriminator
                discriminator_loss = self.discriminator.Discriminator.train_on_batch(x_batch, y_batch)[0]

                # In practice, flipping the label when training the generator improves convergence
                # 用生成的标签训练GAN，并以10%的概率在训练的过程中引入错误的标签
                # 实践中，在训练生成器的过程中翻转标签可以提升收敛性
                if self.flipCoin(chance=0.9):
                    y_generated_labels = np.ones([self.BATCH, 1])
                else:
                    y_generated_labels = np.zeros([self.BATCH, 1])
                x_latent_space_samples = self.sample_latent_space(self.BATCH)
                generator_loss = self.gan.gan_model.train_on_batch(x_latent_space_samples, y_generated_labels)

                # print('Batch: ' + str(int(b)) + ', [Discriminator :: Loss: ' + str(
                #     discriminator_loss) + '], [ Generator :: Loss: ' + str(generator_loss) + ']')
                # if b % self.CHECKPOINT == 0:
                #     label = str(e) + '_' + str(b)
                #     self.plot_checkpoint(label)

            print('Epoch: ' + str(int(e)) + ', [Discriminator :: Loss: ' + str(
                discriminator_loss) + '], [ Generator :: Loss: ' + str(generator_loss) + ']')

            if e % self.CHECKPOINT == 0:
                self.plot_checkpoint(e)
        return

    # 随机生成true/false
    def flipCoin(self, chance=0.5):
        return np.random.binomial(1, chance)
    # 产生潜在空间大小的随机数
    def sample_latent_space(self, instances):
        return np.random.normal(0, 1, (instances, self.LATENT_SPACE_SIZE))

    def plot_checkpoint(self, e):
        filename = "./data/sample_" + str(e) + ".png"

        noise = self.sample_latent_space(16)
        images = self.generator.Generator.predict(noise)

        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
            if self.C == 1:
                image = images[i, :, :]
                image = np.reshape(image, [self.H, self.W])
                image = (255 * (image - np.min(image)) / np.ptp(image)).astype(int)
                plt.imshow(image, cmap='gray')
            elif self.C == 3:
                image = images[i, :, :, :]
                image = np.reshape(image, [self.H, self.W, self.C])
                image = (255 * (image - np.min(image)) / np.ptp(image)).astype(int)
                plt.imshow(image)

            plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
        return

    def plot_check_batch(self, b, images):
        filename = "./data/batch_check_" + str(b) + ".png"
        subplot_size = int(np.sqrt(images.shape[0]))
        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(subplot_size, subplot_size, i + 1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.H, self.W, self.C])
            plt.imshow(image)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
        return
