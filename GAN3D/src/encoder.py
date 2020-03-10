from keras.datasets import mnist
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

# Download the data and format for learning
# 使用.keras文件下的mnist2维数据集，将值映射到0和1之间
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# How much encoding do we want for our setup?
# 创建编码维度——使用多少数字来表示MNIST数据集
# 就是编码后的潜在空间吧
encoding_dimension = 256

# Keras has an input shape of 784
# Keras MNIST数据集尺寸为784，在输入中使用这个值，接下来创建一个编码层和解码层——注意该模型结构十分简单
input_layer = Input(shape=(784,))
encoded_layer = Dense(encoding_dimension, activation='relu')(input_layer)
decoded = Dense(784, activation='sigmoid')(encoded_layer)

# Build the Model
# 这就是创建了一个模型
ac = Model(input_layer, decoded)

# Create an encoder model that we will save later
# 创建编码器模型供以后使用
encoder = Model(input_layer, encoded_layer)

# Train the autoencoder model, ac
# 训练自编码器100epoch，并使用测试数据进行校验
ac.compile(optimizer='adadelta', loss='binary_crossentropy')
ac.fit(x_train, x_train,
       epochs=100,
       batch_size=256,
       shuffle=True,
       validation_data=(x_test, x_test))

# Save the Predicted Data x_train
# 以上就完成了对自编码器的训练，使用自编码器模型来预测整个x_train数据集的编码————在训练结束后保存模型
x_train_encoded = encoder.predict(x_train)
np.save('./x_train_encoded.npy', x_train_encoded)
# Save the Predicted Data x_test
x_test_encoded = encoder.predict(x_test)
np.save('./x_test_encoded.npy', x_test_encoded)
# Save the Encoder model
# 为了复用这个编码器，将模型数据保存成h5文件
encoder.save('./encoder_model.h5')
