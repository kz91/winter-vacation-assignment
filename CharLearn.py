import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from emnist import extract_training_samples, extract_test_samples
import numpy as np
import matplotlib.pyplot as plt

# データセット読み込み
x_train, y_train = extract_training_samples("byclass")
x_test, y_test = extract_test_samples('byclass')

# データ変形
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# 正規化
x_train = x_train / 255.0
x_test = x_test / 255.0

# ラベルの型変換
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("y_train", x_test.shape)
print("y_train", y_test.shape)

np.random.seed(42)
tf.random.set_seed(42)

"""
入力層
"""
model = tf.keras.models.Sequential()

"""
畳み込み層1
フィルタ数：32
フィルタサイズ：3x3
活性化函数：PReLU
"""
model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
model.add(layers.PReLU())
model.add(layers.MaxPooling2D((2, 2)))

"""
畳み込み層2
フィルタ数：64
フィルタサイズ：3x3
活性化函数：PReLU
"""
model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.PReLU())
model.add(layers.MaxPooling2D((2, 2)))

# 平坦化
model.add(layers.Flatten())

# model.add(tf.keras.layers.Dense(128))
# model.add(tf.keras.layers.PReLU())
# model.add(tf.keras.layers.Dropout(0.2))
"""
出力層
ニューロン：62個
活性化函数：Softmax
"""
model.add(tf.keras.layers.Dense(62))
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

model.save('CharModel.h5')
