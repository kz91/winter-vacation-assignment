import tensorflow as tf
import numpy as np
from PIL import Image


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def convert_ps_to_png(ps_path, png_path):
    ps_img = Image.open(ps_path)
    ps_img.save(png_path, "PNG")


def preprocess_image(png_path, target_size=(28, 28)):
    png_img = Image.open(png_path).convert('L')
    png_img = png_img.resize(target_size)
    png_img_array = np.array(png_img) / 255.0
    return np.expand_dims(png_img_array, axis=0)


def predict_character(model, image_array):
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    return predicted_class, prediction


def main():
    # モデルロード
    model_path = 'CharModel.h5'
    model = load_model(model_path)

    # ps2png変換
    ps_path = 'temp.ps'
    png_path = 'temp.png'
    convert_ps_to_png(ps_path, png_path)

    # 画像の正規化
    image_array = preprocess_image(png_path)

    # 予測
    predicted_class, prediction = predict_character(model, image_array)

    # 出力
    all_characters = [chr(i) for i in range(ord('a'), ord('z') + 1)] + \
                      [chr(i) for i in range(ord('A'), ord('Z') + 1)] + \
                      [str(i) for i in range(10)]

    print('Predicted Class:', all_characters[predicted_class])
    print('Prediction Scores:', prediction)


main()