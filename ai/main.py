from pprint import pprint

import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from keras.datasets import mnist
from PIL import Image
from tensorflow.keras.models import load_model
import keras
import keras.layers as layers
from pprint import pprint
from PIL import Image

import numpy as np
# загружаем данные
class Ocr():#по факту это калл с нейронки но что поделавть я бек только делаю
    def __init__(self):
        pass

    def create_model(self):
        try:
            (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
            # меняем формат данных на матрицы 28 на 28
            train_images = train_images.reshape(-1, 28, 28, 1)
            test_images = test_images.reshape(-1, 28, 28, 1)
            train_labels = to_categorical(train_labels)
            test_labels = to_categorical(test_labels)
            import keras
            import keras.layers as layers
            from keras.models import Sequential

            # создаем модель
            model = keras.Sequential()

            # первая свертка (6 фильтров, активация — ReLU)
            model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

            # пулинг (выбираем среднее значение, квадраты 2 на 2)
            model.add(layers.AveragePooling2D(pool_size=2))

            # вторая свертка (16 фильтров, активация ReLU)
            model.add(layers.Conv2D(filters=16, kernel_size=(4, 4), activation='relu'))

            # снова пулинг
            model.add(layers.AveragePooling2D(pool_size=2))

            # превращаем 16 отдельных матриц в один большой вектор
            model.add(layers.Flatten())

            # полносвязные слои

            model.add(layers.Dense(units=120, activation='relu'))

            model.add(layers.Dense(units=10, activation='relu'))

            # выходной слой (активация — SoftMax, для классификации)
            model.add(layers.Dense(units=10, activation='softmax'))

            model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])

            EPOCHS = 4  # количество эпох
            BATCH_SIZE = 32  # размер одного куска данных

            # форматируем входные данные
            train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
            train = train.batch(32)
            test = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
            test = test.batch(32)

            # считаем количество кусков, которые обработаем за одну эпоху (все)
            steps_per_epoch = train_images.shape[0] // BATCH_SIZE
            validation_steps = test_images.shape[0] // BATCH_SIZE

            # обучаем модель
            model.fit(train, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
                      validation_data=test, validation_steps=validation_steps,
                      shuffle=True)
            model.save('my_cnn_model.keras')

            return True
        except Exception:
            return False

    def getImgMatricks(self, srs):
        img = Image.open(f"../base/{srs}")
        # 2. Меняем размер на 128x128 и переводим в ЧБ (режим 'L' - градации серого)
        img = img.resize((28, 28)).convert('L')

        # 3. Преобразуем в матрицу (массив NumPy)
        matrix = np.array(img)
        return matrix // 255

    def pred(self, img):
        img_matrix = np.expand_dims(self.getImgMatricks(img), axis=(0, -1))
        model = load_model("my_cnn_model.keras")
        return np.argmax(model.predict(img_matrix), axis=1)


