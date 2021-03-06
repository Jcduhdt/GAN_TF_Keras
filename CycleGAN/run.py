#!/usr/bin/env python3
from CycleGAN.train import Trainer
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
HEIGHT = 64
WIDTH = 64
CHANNEL = 3
EPOCHS = 100
BATCH = 1
CHECKPOINT = 100
TRAIN_PATH_A = "E:/dataset/horse2zebra/testA/"
TRAIN_PATH_B = "E:/dataset/horse2zebra/trainB/"
TEST_PATH_A = "E:/dataset/horse2zebra/testA/"
TEST_PATH_B = "E:/dataset/horse2zebra/testB/"

trainer = Trainer(height=HEIGHT, width=WIDTH, epochs=EPOCHS, \
                  batch=BATCH, \
                  checkpoint=CHECKPOINT, \
                  train_data_path_A=TRAIN_PATH_A, \
                  train_data_path_B=TRAIN_PATH_B, \
                  test_data_path_A=TEST_PATH_A, \
                  test_data_path_B=TEST_PATH_B, \
                  lambda_cycle=10.0, \
                  lambda_id=1.0)
trainer.train()
