#!/usr/bin/env python3
from GAN3D.src.train import Trainer
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
# 配置显存按需增长
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# Command Line Argument Method
CUBE_SIDE = 16
EPOCHS = 100001
BATCH = 64
CHECKPOINT = 1000
LATENT_SPACE_SIZE = 256
DATA_DIR = "E:/dataset/3D-MNIST/full_dataset_vectors.h5"

trainer = Trainer(side=CUBE_SIDE, \
                  latent_size=LATENT_SPACE_SIZE, \
                  epochs=EPOCHS, \
                  batch=BATCH, \
                  checkpoint=CHECKPOINT, \
                  data_dir=DATA_DIR)
trainer.train()
