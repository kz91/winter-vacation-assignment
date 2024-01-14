import tensorflow as tf
import numpy as np
from PIL import Image
model = tf.keras.models.load_model('charmodel.h5')

ps_img = Image.open('out.ps')
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

print('Predicted Class:', predicted_class)
print('Prediction Scores:', pred)