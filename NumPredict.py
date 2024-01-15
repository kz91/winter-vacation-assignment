import tensorflow as tf
import numpy as np
from PIL import Image

# モデルロード
model = tf.keras.models.load_model('NumModel.h5')

# ps2png変換
ps_img = Image.open('temp.ps')
ps_img.save("out.png", "PNG")

# 画像のリサイズ・正規化
png_img = Image.open('out.png').convert('L')
png_img = png_img.resize((28, 28))
png_img = np.array(png_img)
png_img = png_img / 255.0
png_img = np.expand_dims(png_img, axis=0)

# 予測
pred = model.predict(png_img)
predicted_class = np.argmax(pred)

# 出力
lowercase_letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
uppercase_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
digits = [str(i) for i in range(10)]
all_characters = lowercase_letters + uppercase_letters + digits
print('Predicted Class:', all_characters[predicted_class])
print('Prediction Scores:', pred)