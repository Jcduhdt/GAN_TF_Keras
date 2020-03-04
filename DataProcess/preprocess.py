# -*- coding:utf-8 -*-
# @Time : 2020/2/28 18:15
# @Author : Zhang
# @File : preprocess.py
# @Software: PyCharm
# 演示如何争取读取数

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the UCI Machine Learning Income data from this directory

# Three different ways to read the data
# Notice this incorrectly reads the first line as the header
# 该代码错误将数据的第一行当作头部
df0 = pd.read_csv('E:/tensorflow/data_set/GAN/adult.data')

# The header=None enumerates the classes without a name
# header = None 不适用名字枚举每个类
df1 = pd.read_csv('E:/tensorflow/data_set/GAN/adult.data', header=None)

# The header=None enumerates the classes without a name
# 指定头部，read_csv方法将正确工作
df2 = pd.read_csv('E:/tensorflow/data_set/GAN/adult.data',
                  names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                         'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                         'native-country', 'Label'])

# Create an empty dictionary
mappings = {}

# Run through all columns in the CSV
# 遍历csv文件的所有列
for col_name in df2.columns:
    # If the type of variables are categorical, they will be an 'object' type
    # 如果变量的类型是类别，那么该列数据的类型为Object
    if (df2[col_name].dtype == 'object'):
        # Create a mapping from categorical to numerical variables
        # 创建类别变量和数字的映射
        df2[col_name] = df2[col_name].astype('category')
        df2[col_name], mapping_index = pd.Series(df2[col_name]).factorize()
        # Store the mappings in dictionary
        # 使用字典存储映射关系
        mappings[col_name] = {}
        for i in range(len(mapping_index.categories)):
            mappings[col_name][i] = mapping_index.categories[i]
    # Store a continuous tag for variables that are already numerical
    # 对已经映射过的变量存储一个continuous标签
    else:
        mappings[col_name] = 'continuous'



