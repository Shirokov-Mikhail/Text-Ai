from tensorflow.keras.models import load_model
from PIL import Image
from ai.ai import Ocr

import numpy as np

class Test_A():
     def __init__(self):
         self.model = load_model("../static/my_cnn_model.keras")
         # 1. Открываем фото
         self.img = Image.open("../base/file4.png")
         self.img = self.img.resize((28, 28)).convert('L')# 2. Меняем размер на 128x128 и переводим в ЧБ (режим 'L' - градации серого)
         self.matrix = np.expand_dims(np.array(self.img) // 255, axis=(0, -1))
         # 3. Преобразуем в матрицу (массив NumPy)


     def StartTest(self):
        y_pred = self.model.predict(self.matrix)
        return np.argmax(y_pred, axis=1) == 4

class Test_B():
    def __init__(self):
        self.model = load_model("../static/my_cnn_model.keras")
        # 1. Открываем фото
        self.img = Image.open("../base/file.png")
        self.img = self.img.resize((28, 28)).convert('L')# 2. Меняем размер на 128x128 и переводим в ЧБ (режим 'L' - градации серого)
        self.matrix = np.expand_dims(np.array(self.img) // 255, axis=(0, -1))
        # 3. Преобразуем в матрицу (массив NumPy)


    def StartTest(self):
        y_pred = self.model.predict(self.matrix)
        return np.argmax(y_pred, axis=1) == 8


class Test_C():
    def __init__(self):
        self.model = load_model("../static/my_cnn_model.keras")
        # 1. Открываем фото
        self.img = Image.open("../base/file3.png")
        self.img = self.img.resize((28, 28)).convert('L')# 2. Меняем размер на 128x128 и переводим в ЧБ (режим 'L' - градации серого)
        self.matrix = np.expand_dims(np.array(self.img) // 255, axis=(0, -1))
        # 3. Преобразуем в матрицу (массив NumPy)


    def StartTest(self):
        y_pred = self.model.predict(self.matrix)
        return np.argmax(y_pred, axis=1) == 9

# Теперь 'matrix' — это массив 128x128, где 0 — черный, 255 — белый.

# Пусть img_array — это твоя матрица 28x28


# Нормализуем
#

# Добавляем оси: (batch_size=1, height=28, width=28, channels=1)



# predicted_class = np.argmax(y_pred, axis=1)
# print("Предсказанный класс:", predicted_class[0])
if __name__ == '__main__':
    print(Ocr().pred('f'))
