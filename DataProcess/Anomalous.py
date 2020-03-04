# -*- coding:utf-8 -*-
# @Time : 2020/2/28 21:53
# @Author : Zhang
# @File : Anomalous.py
# @Software: PyCharm
from numpy import linspace, exp
from numpy.random import randn
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

########## Univariate Fit
########## 单变量适配
x = linspace(-5, 5, 200)  # -5~5 200个点
y = exp(-x ** 2) + randn(200) / 10  # 函数加噪
# 对数据使用单变量模型
s = UnivariateSpline(x, y, s=1)  # 啥意思啊 丢数据？
# 定义标准的参数
xs = linspace(-5, 5, 1000)
ys = s(xs)
# 绘制参数
plt.plot(x, y, '.-')
plt.plot(xs, ys)
plt.show()
