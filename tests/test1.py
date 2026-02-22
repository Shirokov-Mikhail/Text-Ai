from tensorflow.keras.models import load_model
import keras
import keras.layers as layers
from pprint import pprint
from PIL import Image

import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
model = load_model("../ai/my_cnn_model.keras")

# 1. Открываем фото
img = Image.open("../base/file4.png")
# 2. Меняем размер на 128x128 и переводим в ЧБ (режим 'L' - градации серого)
img = img.resize((28, 28)).convert('L')

# 3. Преобразуем в матрицу (массив NumPy)
matrix = np.array(img)
pprint(matrix)
# Теперь 'matrix' — это массив 128x128, где 0 — черный, 255 — белый.



img_array = matrix
# Пусть img_array — это твоя матрица 28x28
x = img_array

# Нормализуем
#
x = x//255

# Добавляем оси: (batch_size=1, height=28, width=28, channels=1)
x = np.expand_dims(x, axis=(0, -1))

y_pred = model.predict(x)
print(y_pred)
print(np.argmax(y_pred, axis=1))

# predicted_class = np.argmax(y_pred, axis=1)
# print("Предсказанный класс:", predicted_class[0])