# Документация по распознаванию цифр (DigitRecognizer)

## 📋 Описание

Модуль `ai.py` содержит класс `DigitRecognizer` для распознавания рукописных цифр (0-9) с использованием **Convolutional Neural Network (CNN)** на базе **TensorFlow/Keras**.

Модель обучается на датасете **MNIST**, содержащем 70,000 изображений рукописных цифр:
- **60,000** примеров для обучения
- **10,000** примеров для тестирования

---

## 🏗️ Архитектура модели CNN

```
Input Image (28x28x1)
           ↓
Conv2D (6 фильтров, 3x3) + ReLU
           ↓
AveragePooling2D (2x2)
           ↓
Conv2D (16 фильтров, 4x4) + ReLU
           ↓
AveragePooling2D (2x2)
           ↓
Flatten()
           ↓
Dense (240 нейронов) + ReLU
           ↓
Dense (20 нейронов) + ReLU
           ↓
Dense (10 нейронов) + Softmax
           ↓
Output (вероятности для цифр 0-9)
```

**Параметры модели:**
- Фильтры свертки: 6 и 16
- Функция активации: ReLU (Rectified Linear Unit)
- Пулинг: Average Pooling (2x2)
- Полносвязные слои: 240 → 20 → 10 нейронов
- Выходной слой: Softmax для 10 классов

---

## 🔧 Основные методы

### 1. `__init__(model_path: str | None = None)`

Инициализирует распознаватель цифр.

**Параметры:**
- `model_path` (опционально): Путь к сохраненной модели

**По умолчанию:** модель сохраняется в `ai/my_cnn_model.keras`

```python
from ai.ai import DigitRecognizer

# Инициализация с моделью по умолчанию
recognizer = DigitRecognizer()

# Инициализация с собственным путем
recognizer = DigitRecognizer(model_path="custom_model.keras")
```

---

### 2. `create_model() -> bool`

Создает и обучает новую CNN модель на датасете MNIST.

**Процесс:**
1. Загружает датасет MNIST (70,000 изображений)
2. Нормализует данные (значения 0-1)
3. Переформатирует в формат (samples, 28, 28, 1)
4. Преобразует метки в one-hot кодирование
5. Создает архитектуру CNN
6. Обучает на 60,000 примерах
7. Тестирует на 10,000 примерах
8. Сохраняет модель в `my_cnn_model.keras`

**Возвращает:**
- `True` - если обучение успешно
- `False` - если произошла ошибка

**Пример:**

```python
recognizer = DigitRecognizer()

# Обучение модели (первый раз)
success = recognizer.create_model()

if success:
    print("✅ Модель обучена успешно!")
else:
    print("❌ Ошибка обучения!")
```

**Параметры обучения:**
- Эпохи: 16
- Размер батча: 128
- Оптимизатор: Adam
- Функция потерь: Categorical Crossentropy

**Процесс обучения:**
```
📥 Загрузка датасета MNIST...
✅ Данные загружены: 60000 примеров для обучения
🔨 Создание архитектуры CNN...
📊 Архитектура модели:
[модель будет выведена]
🎓 Начинается обучение модели...
Epoch 1/16
...
Epoch 16/16
💾 Сохранение модели в ai/my_cnn_model.keras...
✅ Обучение завершено успешно!
```

---

### 3. `recognize_digit(image_path: str | Path) -> int`

Распознает одну цифру из изображения.

**Параметры:**
- `image_path`: Путь к файлу изображения (PNG, JPG, BMP, etc.)

**Возвращает:**
- `int`: Распознанная цифра (0-9)

**Процесс:**
1. Загружает изображение
2. Преобразует в черно-белое (28x28)
3. Нормализует значения
4. Пропускает через модель
5. Возвращает цифру с наибольшей вероятностью

**Пример:**

```python
recognizer = DigitRecognizer()

# Распознавание одной цифры
digit = recognizer.recognize_digit("image.png")
print(f"Распознана цифра: {digit}")

# Вывод:
# 🎯 Распознана цифра: 7 (уверенность: 98.5%)
```

---

### 4. `recognize_digits_batch(image_paths: list) -> list`

Распознает цифры в нескольких изображениях.

**Параметры:**
- `image_paths`: Список путей к файлам изображений

**Возвращает:**
- `list`: Список распознанных цифр

**Пример:**

```python
recognizer = DigitRecognizer()

# Распознавание нескольких цифр
images = ["digit1.png", "digit2.png", "digit3.png"]
results = recognizer.recognize_digits_batch(images)

print(results)  # [5, 3, 8]
```

---

### 5. `pred(image_path: str | Path) -> np.ndarray`

Распознает цифру и возвращает результат в виде массива NumPy.

**Назначение:** Обратная совместимость со старым кодом

**Параметры:**
- `image_path`: Путь к изображению

**Возвращает:**
- `np.ndarray`: Массив с распознанной цифрой

**Пример:**

```python
recognizer = DigitRecognizer()
result = recognizer.pred("digit.png")
print(result)  # [7]
```

---

### 6. `result(image_path: str | Path) -> list`

Распознает цифру и возвращает результат в виде списка строк.

**Назначение:** Обратная совместимость со старым кодом

**Параметры:**
- `image_path`: Путь к изображению

**Возвращает:**
- `list`: Список со строковым представлением цифры

**Пример:**

```python
recognizer = DigitRecognizer()
result = recognizer.result("digit.png")
print(result)  # ['7']
```

---

## 📊 Параметры класса

| Параметр | Значение | Описание |
|----------|---------|---------|
| `IMAGE_SIZE` | 28 | Размер входного изображения (28x28 пиксели) |
| `NUM_CLASSES` | 10 | Количество классов (цифры 0-9) |
| `EPOCHS` | 16 | Количество эпох обучения |
| `BATCH_SIZE` | 128 | Размер пакета данных |

---

## 📝 Использование

### Пример 1: Обучение модели

```python
from ai.ai import DigitRecognizer

# Создаем экземпляр распознавателя
recognizer = DigitRecognizer()

# Обучаем модель
success = recognizer.create_model()

if success:
    print("✅ Модель обучена!")
```

### Пример 2: Распознавание одной цифры

```python
from ai.ai import DigitRecognizer

recognizer = DigitRecognizer()

# Распознаем цифру
digit = recognizer.recognize_digit("path/to/digit.png")
print(f"Результат: {digit}")
```

### Пример 3: Распознавание нескольких цифр

```python
from ai.ai import DigitRecognizer

recognizer = DigitRecognizer()

# Список изображений
images = ["0.png", "1.png", "2.png"]

# Распознаем все
results = recognizer.recognize_digits_batch(images)
print(results)  # [0, 1, 2]
```

### Пример 4: Интеграция с Flask приложением

```python
from flask import Flask
from ai.ai import DigitRecognizer

app = Flask(__name__)
recognizer = DigitRecognizer()

@app.route('/recognize/<filename>')
def recognize(filename):
    digit = recognizer.recognize_digit(f"base/{filename}")
    return {"digit": digit}
```

---

## 🎯 Подготовка изображения

Требования к входному изображению:

1. **Размер:** Любой (будет автоматически изменен на 28x28)
2. **Цвет:** Любой (будет преобразовано в черно-белое)
3. **Формат:** PNG, JPG, BMP, GIF и т.д.
4. **Содержимое:** Одна четкая цифра

**Метод `_prepare_image()`:**

```python
@staticmethod
def _prepare_image(image_path: str | Path) -> np.ndarray:
    # 1. Открытие и преобразование в ЧБ
    image = Image.open(image_path).convert('L')
    
    # 2. Изменение размера на 28x28
    image = image.resize((28, 28), Image.LANCZOS)
    
    # 3. Преобразование в матрицу и нормализация (0-1)
    matrix = np.array(image, dtype=np.float32) / 255.0
    
    # 4. Инверсия если нужно
    if matrix.mean() > 0.5:
        matrix = 1.0 - matrix
    
    return matrix
```

---

## 🚀 Запуск как скрипт

Для обучения модели напрямую:

```bash
cd ai/
python ai.py
```

**Вывод:**

```
🚀 Модель не найдена. Начинается обучение...
📥 Загрузка датасета MNIST...
✅ Данные загружены: 60000 примеров для обучения
🔨 Создание архитектуры CNN...
[...]
✅ Обучение завершено успешно!
```

---

## ⚠️ Обработка ошибок

### Ошибка 1: Модель не найдена

```
FileNotFoundError: Модель не найдена: ai/my_cnn_model.keras
Сначала обучите модель, запустив: create_model()
```

**Решение:**

```python
recognizer = DigitRecognizer()
recognizer.create_model()  # Обучить модель
```

### Ошибка 2: Ошибка при обучении

```
❌ Ошибка при обучении модели: [причина]
```

**Решение:**
- Проверьте наличие памяти
- Проверьте установку TensorFlow
- Проверьте CUDA поддержку (если используется GPU)

### Ошибка 3: Файл изображения не найден

```
FileNotFoundError: [Errno 2] No such file or directory
```

**Решение:**
- Убедитесь в правильности пути
- Используйте абсолютные пути

---

## 📈 Производительность

**Время обучения:**
- На CPU: 10-20 минут
- На GPU (CUDA): 2-5 минут

**Точность модели:**
- На тестовом наборе: ~99%
- На реальных изображениях: 90-95%

**Время распознавания:**
- На CPU: 0.1-0.3 сек
- На GPU: 0.05-0.1 сек

---

## 🔄 Отличия от старого кода

| Параметр | Старо | Ново |
|----------|------|------|
| Имя класса | `Ocr` | `DigitRecognizer` |
| Размер изображения | Переменный | Фиксированный 28x28 |
| Датасет | Не полный | Полный MNIST |
| Документация | Отсутствует | Подробная |
| Обработка ошибок | Плохая | Хорошая |
| Типизация | Нет | Да (type hints) |
| Архитектура | 1 свертка | 2 свертки |

---

## 📚 Дополнительные ресурсы

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/api/)
- [CNN Tutorial](https://www.ibm.com/cloud/learn/convolutional-neural-networks)

---

**Версия:** 1.0  
**Последнее обновление:** 2026-04-06  
**Автор:** Text-Ai Team

