#!/usr/bin/env python3
from DCGAN.train import Trainer
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
# 配置显存按需增长
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# Command Line Argument Method
HEIGHT = 96
WIDTH = 96
CHANNEL = 3
LATENT_SPACE_SIZE = 100
EPOCHS = 100
BATCH = 64
CHECKPOINT = 10
PATH = "./data/face_color.npy"

trainer = Trainer(height=HEIGHT, \
                  width=WIDTH, \
                  channels=CHANNEL, \
                  latent_size=LATENT_SPACE_SIZE, \
                  epochs=EPOCHS, \
                  batch=BATCH, \
                  checkpoint=CHECKPOINT, \
                  model_type='DCGAN', \
                  data_path=PATH)

trainer.train()
