#!/usr/bin/env python3
from FullGAN.gan import GAN
from FullGAN.generator import Generator
from FullGAN.discriminator import Discriminator
# 是从网站上下载数据集到.keras/dataset目录下读取，若该路径有就不下载了
# 下载的是mnist.npz文件,在百度云有
from keras.datasets import mnist
from random import randint
import numpy as np
import matplotlib.pyplot as plt


class Trainer:
    # 初始化 对GAN进行调优的变量以及方法本身的初始化
    def __init__(self, width=28, height=28, channels=1, latent_size=100, epochs=50000, batch=32, checkpoint=50,
                 model_type=-1):
        self.W = width
        self.H = height
        self.C = channels
        # 训练的次数
        self.EPOCHS = epochs
        # batch_size 每次训练的图片数
        self.BATCH = batch
        self.CHECKPOINT = checkpoint
        # model_type =-1 即使用0-9的整体进行训练，=0-9 即使用单独的一个数字进行训练
        self.model_type = model_type
        self.LATENT_SPACE_SIZE = latent_size

        # 初始化GAN的Generator、Discriminator、GAN
        self.generator = Generator(height=self.H, width=self.W, channels=self.C, latent_size=self.LATENT_SPACE_SIZE)
        self.discriminator = Discriminator(height=self.H, width=self.W, channels=self.C)
        self.gan = GAN(generator=self.generator.Generator, discriminator=self.discriminator.Discriminator)
        # 载入数据集到类中
        self.load_MNIST()

    def load_MNIST(self, model_type=3):
        allowed_types = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if self.model_type not in allowed_types:
            print('ERROR: Only Integer Values from -1 to 9 are allowed')

        # 读取.kersa/dataset/mnist.npz文件，将训练集和数据集读入
        # Y_train只有在想选择某个特定数字生成模型时使用
        (self.X_train, self.Y_train), (_, _) = mnist.load_data()
        # 如果model_type != -1，即使用特定数字的图片进行训练，使X_train是该数字的训练集
        if self.model_type != -1:
            self.X_train = self.X_train[np.where(self.Y_train == int(self.model_type))[0]]

        # Rescale -1 to 1
        # Find Normalize Function from CV Class
        # 这是啥
        self.X_train = (np.float32(self.X_train) - 127.5) / 127.5
        self.X_train = np.expand_dims(self.X_train, axis=3)
        return

    # 训练方法
    def train(self):
        # 训练次数
        for e in range(self.EPOCHS):
            # Train Discriminator
            # 令每次训练时的bathc一半是真的一般是噪声
            # Make the training batch for this model be half real, half noise
            # Grab Real Images for this training batch
            # 获取一个batch的数据
            count_real_images = int(self.BATCH / 2)
            # 随机获得0到（训练集长度-要获取长度）的整数下标
            # 因为下一步是从训练集的这个下标开始连续的要获取的16个数据
            starting_index = randint(0, (len(self.X_train) - count_real_images))
            real_images_raw = self.X_train[starting_index: (starting_index + count_real_images)]
            # 将这16个图片数据转换成16*28*28*1的数据输入
            x_real_images = real_images_raw.reshape(count_real_images, self.W, self.H, self.C)
            # 令真实数据的标签为1
            y_real_labels = np.ones([count_real_images, 1])

            # Grab Generated Images for this training batch
            # 获取生成器的产生数据
            latent_space_samples = self.sample_latent_space(count_real_images)
            # predict即生成器模型的输出
            x_generated_images = self.generator.Generator.predict(latent_space_samples)
            # 令生成器生成的图像标签为0
            y_generated_labels = np.zeros([self.BATCH - count_real_images, 1])

            # Combine to train on the discriminator
            # 对每个批次的训练数据和训练标签进行拼接
            x_batch = np.concatenate([x_real_images, x_generated_images])
            y_batch = np.concatenate([y_real_labels, y_generated_labels])

            # Now, train the discriminator with this batch
            # 训练discriminator模型 获得训练状态loss
            # 训练过程中判别器会知道图片是否伪造，一直发现和真实图片相比不完美的地方
            discriminator_loss = self.discriminator.Discriminator.train_on_batch(x_batch, y_batch)[0]

            # Generate Noise
            # 产生噪声
            x_latent_space_samples = self.sample_latent_space(self.BATCH)
            y_generated_labels = np.ones([self.BATCH, 1])
            # 使用生成器错误标记的输出训练GAN，也就是说从噪声中产生图像，并在GAN训练过程中给图像指定一个标签
            # 即对抗部分，使用新训练的判别器提升生成输出-GAN的损失函数将会描述判别器对生成结果所产生的困惑
            # 因为在GAN这个大类里，discriminator是不训练的，只是当作判别使用，在训练GAN之前就已经不停训练了discriminator
            # 所以若重新使用生成器生成图像并令其标签正确，那么整个GAN网络就会优化generator，使其生成得更逼真
            # 训练生成器获得loss
            generator_loss = self.gan.gan_model.train_on_batch(x_latent_space_samples, y_generated_labels)

            if e % self.CHECKPOINT == 0:
                # 将损失指标输出到屏幕
                print('Epoch: ' + str(int(e)) + ', [Discriminator :: Loss: ' + str(
                    discriminator_loss) + '], [ Generator :: Loss: ' + str(generator_loss) + ']')
                # 每CHECKPOINT次将当前图像输出到data目录下
                self.plot_checkpoint(e)
        return

    # 产生随机数据，instance为每个批次要产生的数量，LATENT_SPACE_SIZE即输入维度128
    def sample_latent_space(self, instances):
        return np.random.normal(0, 1, (instances, self.LATENT_SPACE_SIZE))

    # 保存图像
    def plot_checkpoint(self, e):
        filename = "./data/sample_" + str(e) + ".png"

        # 生成当前训练次数下模型的16张图
        noise = self.sample_latent_space(16)
        images = self.generator.Generator.predict(noise)

        # 创建自定义图像，figsize:指定figure的宽和高，单位为英寸
        plt.figure(figsize=(10, 10))
        # 将一张图分成4*4的大小拼接16张图
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.H, self.W])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        # tight_layout会自动调整子图参数，使之填充整个图像区域
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
        return
