#!/usr/bin/env python3
from SimGAN.train import Trainer
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
HEIGHT = 55
WIDTH = 35
CHANNELS = 1
EPOCHS = 100
BATCH = 16
CHECKPOINT = 50
SIM_PATH = "E:/dataset/eye_gaze/gaze.h5"
REAL_PATH = "E:/dataset/eye_gaze/real_gaze.h5"

trainer = Trainer(height=HEIGHT, width=WIDTH, channels=CHANNELS, epochs=EPOCHS, \
                  batch=BATCH, \
                  checkpoint=CHECKPOINT, \
                  sim_path=SIM_PATH, \
                  real_path=REAL_PATH)
trainer.train()
