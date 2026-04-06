"""
Модуль для распознавания цифр (0-9) с использованием CNN (Convolutional Neural Network).
Использует датасет MNIST для обучения и TensorFlow/Keras для создания модели.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


class DigitRecognizer:
    """
    Класс для распознавания цифр (0-9) с помощью сверточной нейронной сети (CNN).

    Использует датасет MNIST: 70000 изображений цифр размером 28x28 пиксели.
    Модель обучается на 60000 примерах и тестируется на 10000 примерах.
    """

    # Параметры модели
    IMAGE_SIZE = 28  # Размер входного изображения (28x28 пиксели)
    NUM_CLASSES = 10  # Количество классов (цифры 0-9)
    EPOCHS = 16  # Количество эпох обучения
    BATCH_SIZE = 128  # Размер пакета данных

    def __init__(self, model_path: str | None = None):
        """
        Инициализирует распознаватель цифр.

        Args:
            model_path: Путь к сохраненной модели (опционально)
        """
        default_path = Path(__file__).resolve().parent / "my_cnn_model.keras"
        self.model_path = Path(model_path) if model_path else default_path
        self.model = None

    def create_model(self) -> bool:
        """
        Создает и обучает CNN модель для распознавания цифр.

        Архитектура модели:
        - Conv2D (6 фильтров, 3x3) → AveragePooling2D
        - Conv2D (16 фильтров, 4x4) → AveragePooling2D
        - Flatten
        - Dense (240 нейронов) → ReLU
        - Dense (20 нейронов) → ReLU
        - Dense (10 нейронов) → SoftMax (выход: вероятности для цифр 0-9)

        Returns:
            bool: True если обучение успешно, False если произошла ошибка
        """
        try:
            print("📥 Загрузка датасета MNIST...")
            # Загружаем датасет MNIST
            (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

            # Нормализуем данные (значения от 0 до 1)
            train_images = train_images.astype('float32') / 255.0
            test_images = test_images.astype('float32') / 255.0

            # Переформатируем в формат (samples, height, width, channels)
            train_images = train_images.reshape(-1, 28, 28, 1)
            test_images = test_images.reshape(-1, 28, 28, 1)

            # Преобразуем метки в one-hot кодирование
            train_labels = to_categorical(train_labels, self.NUM_CLASSES)
            test_labels = to_categorical(test_labels, self.NUM_CLASSES)

            print(f"✅ Данные загружены: {train_images.shape[0]} примеров для обучения")

            # Создаем CNN модель
            print("🔨 Создание архитектуры CNN...")
            model = models.Sequential([
                # Первый сверточный блок
                layers.Conv2D(
                    filters=6,
                    kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 1)
                ),
                layers.AveragePooling2D(pool_size=2),

                # Второй сверточный блок
                layers.Conv2D(
                    filters=16,
                    kernel_size=(4, 4),
                    activation='relu'
                ),
                layers.AveragePooling2D(pool_size=2),

                # Flatten слой
                layers.Flatten(),

                # Полносвязные слои
                layers.Dense(units=240, activation='relu'),
                layers.Dense(units=20, activation='relu'),

                # Выходной слой (10 классов для цифр 0-9)
                layers.Dense(units=self.NUM_CLASSES, activation='softmax')
            ])

            # Компилируем модель
            model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )

            print("📊 Архитектура модели:")
            model.summary()

            # Подготавливаем данные для обучения
            print("🎓 Начинается обучение модели...")
            train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
            train_dataset = train_dataset.batch(self.BATCH_SIZE).shuffle(buffer_size=10000)

            test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
            test_dataset = test_dataset.batch(self.BATCH_SIZE)

            # Обучаем модель
            history = model.fit(
                train_dataset,
                epochs=self.EPOCHS,
                validation_data=test_dataset,
                verbose=1
            )

            # Сохраняем модель
            print(f"💾 Сохранение модели в {self.model_path}...")
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(self.model_path)
            self.model = model

            print("✅ Обучение завершено успешно!")
            return True

        except Exception as e:
            print(f"❌ Ошибка при обучении модели: {e}")
            return False

    def _ensure_model(self) -> None:
        """Загружает модель, если она еще не загружена."""
        if self.model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Модель не найдена: {self.model_path}\n"
                f"Сначала обучите модель, запустив: create_model()"
            )

        print(f"📂 Загрузка модели из {self.model_path}...")
        self.model = load_model(self.model_path)

    @staticmethod
    def _prepare_image(image_path: str | Path) -> np.ndarray:
        """
        Подготавливает изображение для распознавания.

        Args:
            image_path: Путь к изображению

        Returns:
            np.ndarray: Нормализованная матрица (28x28, значения 0-1)
        """
        # Открываем изображение и преобразуем в ЧБ
        image = Image.open(image_path).convert('L')

        # Изменяем размер на 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # Преобразуем в матрицу NumPy и нормализуем
        matrix = np.array(image, dtype=np.float32) / 255.0

        # Инвертируем цвета если нужно (белый текст на черном фоне)
        if matrix.mean() > 0.5:
            matrix = 1.0 - matrix

        return matrix

    def recognize_digit(self, image_path: str | Path) -> int:
        """
        Распознает цифру на изображении.

        Args:
            image_path: Путь к изображению с цифрой

        Returns:
            int: Распознанная цифра (0-9)
        """
        self._ensure_model()

        # Подготавливаем изображение
        image_matrix = self._prepare_image(image_path)

        # Добавляем размер батча и канала
        image_input = np.expand_dims(image_matrix, axis=(0, -1))

        # Предсказываем
        prediction = self.model.predict(image_input, verbose=0)

        # Возвращаем цифру с наибольшей вероятностью
        digit = np.argmax(prediction[0])
        confidence = prediction[0][digit]

        print(f"🎯 Распознана цифра: {digit} (уверенность: {confidence:.2%})")
        return int(digit)

    def recognize_digits_batch(self, image_paths: list) -> list:
        """
        Распознает цифры в нескольких изображениях.

        Args:
            image_paths: Список путей к изображениям

        Returns:
            list: Список распознанных цифр
        """
        self._ensure_model()

        results = []
        for image_path in image_paths:
            try:
                digit = self.recognize_digit(image_path)
                results.append(digit)
            except Exception as e:
                print(f"❌ Ошибка обработки {image_path}: {e}")
                results.append(None)

        return results

    # Обратная совместимость с методами из старого кода
    def pred(self, image_path: str | Path) -> np.ndarray:
        """
        Распознает цифру и возвращает массив (обратная совместимость).

        Args:
            image_path: Путь к изображению

        Returns:
            np.ndarray: Массив с распознанной цифрой
        """
        digit = self.recognize_digit(image_path)
        return np.array([digit], dtype=np.int64)

    def result(self, image_path: str | Path) -> list:
        """
        Распознает цифру и возвращает результат в виде списка (обратная совместимость).

        Args:
            image_path: Путь к изображению

        Returns:
            list: Список с распознанной цифрой в виде строки
        """
        digit = self.recognize_digit(image_path)
        return [str(digit)]


if __name__ == "__main__":
    # Пример использования
    recognizer = DigitRecognizer()

    # Проверяем существует ли уже обученная модель
    if not recognizer.model_path.exists():
        print("🚀 Модель не найдена. Начинается обучение...")
        success = recognizer.create_model()
        if not success:
            print("❌ Ошибка при обучении модели!")
    else:
        print(f"✅ Модель найдена: {recognizer.model_path}")
        print("Для обучения новой модели удалите файл:", recognizer.model_path)
