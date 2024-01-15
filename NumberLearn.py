import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

# データセット読み込み
mnist = tf.keras.datasets.mnist.load_data()
train, test = mnist
(x_train, y_train),(x_test, y_test) = mnist

print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("y_train", x_test.shape)
print("y_train", y_test.shape)

# 正規化
x_train = x_train / 255.0
x_test = x_test / 255.0

"""
入力層
28x28の入力を1x784の配列に変換
"""
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input((28, 28)))
model.add(tf.keras.layers.Flatten())

"""
中間層
ニューロン：128個(全結合）
活性化函数：ReLu
ドロップアウト：20%
"""
model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
model.add(tf.keras.layers.Dropout(0.2))

"""
出力層
ニューロン：10個
活性化函数：Softmax
"""
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation(tf.keras.activations.softmax))

model.compile(
   optimizer=tf.keras.optimizers.Adam(),
   loss=tf.keras.losses.sparse_categorical_crossentropy,
   metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)

# 早期終了
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(
    x_train, y_train,
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping]
)

model.save('NumModel.h5')
