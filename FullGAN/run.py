#!/usr/bin/env python3
from FullGAN.train import Trainer

# Command Line Argument Method
# 根据图像数据写得
HEIGHT = 28
WIDTH = 28
CHANNEL = 1
# 潜在空间尺寸，即使用多长的向量生成图片
LATENT_SPACE_SIZE = 100
# 训练次数
EPOCHS = 50001
# batch_size
BATCH = 32
# 每这么多次，打印当前loss保存当前生成模型产生的图片
CHECKPOINT = 500
# 取值=-1即将0-9的数据进行训练，取值0-9即单独训练某一个数
MODEL_TYPE = -1

trainer = Trainer(height=HEIGHT, \
                  width=WIDTH, \
                  channels=CHANNEL, \
                  latent_size=LATENT_SPACE_SIZE, \
                  epochs=EPOCHS, \
                  batch=BATCH, \
                  checkpoint=CHECKPOINT,
                  model_type=MODEL_TYPE)

trainer.train()
