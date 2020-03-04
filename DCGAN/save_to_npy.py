#!/usr/bin/env python3
from PIL import Image
import numpy as np
import os


# 获取指定目录下特定扩展名文件的列表
# 递归搜索一个目录下所有有该扩展名的文件
# 该例子中，所有LSUN数据集中的图像都被保存成WenP格式
# def grabListOfFiles(startingDirectory, extension=".webp"):
def grabListOfFiles(startingDirectory, extension=".jpg"):
    listOfFiles = []
    for file in os.listdir(startingDirectory):
        if file.endswith(extension):
            listOfFiles.append(os.path.join(startingDirectory, file))
    return listOfFiles


# 载入图像，并根据参数将图像转换成RGB或者是灰度的格式
# 使用Pillow中的Image类可以正确读取WebP格式的文件。通过pillow的内置功能，可以将图像转换成更小尺寸（合理尺寸64*64）
# 图像被加载和转换后，见加入一个数组。处理完所有文件后，返回该数组
def grabArrayOfImages(listOfFiles, resizeW=96, resizeH=96, gray=False):
    imageArr = []
    for f in listOfFiles:
        if gray:
            im = Image.open(f).convert("L")
        else:
            im = Image.open(f).convert("RGB")
        im = im.resize((resizeW, resizeH))
        imData = np.asarray(im)
        imageArr.append(imData)
    return imageArr


# 调用上面两种函数，获取文件，转换图像，使用Numpy内置save函数将图像保存成npy格式
direc = "E:/dataset/GAN_face/faces/"
listOfFiles = grabListOfFiles(direc)
imageArrGray = grabArrayOfImages(listOfFiles, resizeW=96, resizeH=96, gray=True)
imageArrColor = grabArrayOfImages(listOfFiles, resizeW=96, resizeH=96)
print("Shape of ImageArr Gray: ", np.shape(imageArrGray))
print("Shape of ImageArr Color: ", np.shape(imageArrColor))
np.save('./data/face_gray.npy', imageArrGray)
np.save('./data/face_color.npy', imageArrColor)
