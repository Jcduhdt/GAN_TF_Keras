#!/usr/bin/env python3
from Pix2Pix.gan import GAN
from Pix2Pix.generator import Generator
from Pix2Pix.discriminator import Discriminator
from keras.layers import Input
from keras.datasets import mnist
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os
from PIL import Image
import random
import numpy as np


class Trainer:
    def __init__(self, height=256, width=256, channels=3, epochs=50000, batch=1, checkpoint=50, train_data_path='',
                 test_data_path=''):
        self.EPOCHS = epochs
        self.BATCH = batch
        self.H = height
        self.W = width
        self.C = channels
        self.CHECKPOINT = checkpoint

        # 载入训练所需要的数据集以及另一组之后绘图用来比较的数据集
        self.X_train_B, self.X_train_A = self.load_data(train_data_path)
        self.X_test_B, self.X_test_A = self.load_data(test_data_path)
        # 初始化Generator对象
        self.generator = Generator(height=self.H, width=self.W, channels=self.C)
        # 创建两个和输入图像尺寸、形状以及通道数都完全相同的输入
        self.orig_A = Input(shape=(self.W, self.H, self.C))
        self.orig_B = Input(shape=(self.W, self.H, self.C))
        # 条件训练的部分--使用orig_B作为生成器的输入
        self.fake_A = self.generator.Generator(self.orig_B)
        # 构建Discriminator对象，并将判别器网络的可训练性设置为False
        self.discriminator = Discriminator(height=self.H, width=self.W, channels=self.C)
        self.discriminator.trainable = False
        # 将fake_A和orig_B作为输入传递给discriminator对象
        self.valid = self.discriminator.Discriminator([self.fake_A, self.orig_B])
        # 最后，做好了所有连接配置
        # 使用输入orig_A和orig_B以及输出校验和fake_A来创建对抗网络
        model_inputs = [self.orig_A, self.orig_B]
        model_outputs = [self.valid, self.fake_A]
        self.gan = GAN(model_inputs=model_inputs, model_outputs=model_outputs)

    # 数据加载函数定义了需要加载的数据--所有图像都被连接到一张256*512的图像中，因此这个函数需要载入所有图像并将其切分成两个数组
    def load_data(self, data_path):
        listOFFiles = self.grabListOfFiles(data_path, extension="jpg")
        imgs_temp = np.array(self.grabArrayOfImages(listOFFiles))
        imgs_A = []
        imgs_B = []
        for img in imgs_temp:
            # imgs_A存储的是256*512的前一半图
            # imgs_B存储的是256*512的后一半图
            # 图的形状好像没有想清楚，还要仔细想想
            imgs_A.append(img[:, :self.H])
            imgs_B.append(img[:, self.H:])

        imgs_A_out = self.norm_and_expand(np.array(imgs_A))
        imgs_B_out = self.norm_and_expand(np.array(imgs_B))

        return imgs_A_out, imgs_B_out

    # 帮助将数组调整成网络可以使用的格式
    def norm_and_expand(self, arr):
        arr = (arr.astype(np.float32) - 127.5) / 127.5
        normed = np.expand_dims(arr, axis=3)
        return normed

    # 从目录中获取文件列表
    def grabListOfFiles(self, startingDirectory, extension=".webp"):
        listOfFiles = []
        for file in os.listdir(startingDirectory):
            if file.endswith(extension):
                listOfFiles.append(os.path.join(startingDirectory, file))
        return listOfFiles

    # 根据已有的文件列表，将图像加载进数组并返回
    def grabArrayOfImages(self, listOfFiles, gray=False):
        imageArr = []
        for f in listOfFiles:
            if gray:
                im = Image.open(f).convert("L")
            else:
                im = Image.open(f).convert("RGB")
            imData = np.asarray(im)
            imageArr.append(imData)
        return imageArr

    def train(self):
        for e in range(self.EPOCHS):
            # 定义一个在多次epoch中需要反复掉哟个的训练方法——在每次epoch中，都需要幅值训练数据AB
            b = 0
            # 真实图片
            X_train_A_temp = deepcopy(self.X_train_A)
            # 真实图片对应的风格图片
            X_train_B_temp = deepcopy(self.X_train_B)

            full_loss = 0
            generator_loss = 0
            # 定义batch的数量(对于风格转换类型，batch大小为1)并运行batch
            # batch大小为1，但是每个epoch训练数据集大小的图片数
            number_of_batches = len(self.X_train_A)
            for b in range(number_of_batches):
                # Train Discriminator
                # Grab Real Images for this training batch
                # 随机选取A和B中的索引（数据是成对出现的，因此需要使用A和B中相同的索引）
                # 随机选一张图
                starting_ind = randint(0, (len(X_train_A_temp) - 1))
                real_images_raw_A = X_train_A_temp[starting_ind: (starting_ind + 1)]
                real_images_raw_B = X_train_B_temp[starting_ind: (starting_ind + 1)]

                # Delete the images used until we have none left
                # 选取图像后，从临时数组中删除这些图像
                # 删除使用过的图像，直到临时数组变空
                X_train_A_temp = np.delete(X_train_A_temp, range(starting_ind, (starting_ind + 1)), 0)
                X_train_B_temp = np.delete(X_train_B_temp, range(starting_ind, (starting_ind + 1)), 0)
                # 创建batch来进行预测和训练
                batch_A = real_images_raw_A.reshape(1, self.W, self.H, self.C)
                batch_B = real_images_raw_B.reshape(1, self.W, self.H, self.C)

                # PatchGAN
                # 根据PatchGAN论文中的架构来构建Y标签，创建y_valid和y_fake标签进行训练
                # 形状(1,16,16,1)
                y_valid = np.ones((1,) + (int(self.W / 2 ** 4), int(self.W / 2 ** 4), 1))
                y_fake = np.zeros((1,) + (int(self.W / 2 ** 4), int(self.W / 2 ** 4), 1))
                # 根据batch_B得输入使用生成器来生成FakeA图像
                fake_A = self.generator.Generator.predict(batch_B)

                # Now, train the discriminator with this batch of reals
                # 使用真实的数据和伪造的数据来训练判别器——并记录每次迭代两者的损失
                discriminator_loss_real = self.discriminator.Discriminator.train_on_batch([batch_A, batch_B], y_valid)[
                    0]
                discriminator_loss_fake = self.discriminator.Discriminator.train_on_batch([fake_A, batch_B], y_fake)[0]
                # 通过平均两个损失值得到一个聚合损失
                full_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
                # 使用batch_A和batch_B作为输入。并以y_valid和batch_A作为输出训练对抗模型
                generator_loss = self.gan.gan_model.train_on_batch([batch_A, batch_B], [y_valid, batch_A])

                print('Batch: ' + str(int(b)) + ', [Full Discriminator :: Loss: ' + str(
                    full_loss) + '], [ Generator :: Loss: ' + str(generator_loss) + ']')
                if b % self.CHECKPOINT == 0:
                    label = str(e) + '_' + str(b)
                    self.plot_checkpoint(label)

            print('Epoch: ' + str(int(e)) + ', [Full Discriminator :: Loss:' + str(
                full_loss) + '], [ Generator :: Loss: ' + str(generator_loss) + ']')

        return

    def plot_checkpoint(self, b):
        orig_filename = "./out/batch_check_" + str(b) + "_original.png"

        r, c = 3, 3
        random_inds = random.sample(range(len(self.X_test_A)), 3)
        imgs_A = self.X_test_A[random_inds].reshape(3, self.W, self.H, self.C)
        imgs_B = self.X_test_B[random_inds].reshape(3, self.W, self.H, self.C)
        fake_A = self.generator.Generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Style', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("./out/batch_check_" + str(b) + ".png")
        plt.close('all')

        return
