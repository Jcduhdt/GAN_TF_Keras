#!/usr/bin/env python3
from Pix2Pix.train import Trainer
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
# 配置显存按需增长
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
set_session(tf.Session(config=config))

# Command Line Argument Method
HEIGHT = 256
WIDTH = 256
CHANNELS = 3
EPOCHS = 100
BATCH = 1
CHECKPOINT = 50
TRAIN_PATH = "E:/dataset/cityscapes/train/"
TEST_PATH = "E:/dataset/cityscapes/val/"

trainer = Trainer(height=HEIGHT, width=WIDTH, channels=CHANNELS, epochs=EPOCHS, \
                  batch=BATCH, \
                  checkpoint=CHECKPOINT, \
                  train_data_path=TRAIN_PATH, \
                  test_data_path=TEST_PATH)
trainer.train()
