from PIL import Image
ps_img = Image.open('out.ps')
ps_img.resize((28, 28))
ps_img.save("out.png", "PNG")

png_img = Image.open('out.png')
png_img = png_img / 255.0

img_array = np.expand_dims(png_img, axis=0)
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
print('Predicted Class:', predicted_class)
