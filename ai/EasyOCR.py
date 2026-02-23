import easyocr
import cv2
import torch
from flask import url_for


# Проверяем, видит ли Python твою видеокарту


class Ocr():
    def __init__(self):
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version PyTorch expects: {torch.version.cuda}")
        print(f"CUDA доступна: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
             print(f"Используется: {torch.cuda.get_device_name(0)}")
        self.reader = easyocr.Reader(['ru', 'en'], gpu=True)

    def result(self, img):
        result = self.reader.readtext(f'base/{img}', detail=0)
        # Печатаем результат
        return result

if __name__ == '__main__':
    print(Ocr().result('textFile.png'))
