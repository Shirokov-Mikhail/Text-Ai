import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ai.ai import Ocr


class DummyModel:
    def __init__(self, predicted_indices: list[int], class_count: int):
        self.predicted_indices = predicted_indices
        self.class_count = class_count

    def predict(self, batch, verbose=0):
        out = np.zeros((len(self.predicted_indices), self.class_count), dtype=np.float32)
        for i, idx in enumerate(self.predicted_indices):
            out[i, idx] = 1.0
        return out


class OcrTests(unittest.TestCase):
    def test_prepare_image_inverts_light_background(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "img.png"
            img = Image.new("L", (20, 20), color=255)
            draw = ImageDraw.Draw(img)
            draw.rectangle((8, 8, 12, 12), fill=0)
            img.save(img_path)

            matrix = Ocr._prepare_image(img_path)
            self.assertGreater(matrix[10, 10], matrix[0, 0])

    def test_split_symbols_returns_two_symbols(self):
        matrix = np.zeros((48, 60), dtype=np.float32)
        matrix[8:20, 4:12] = 1.0
        matrix[8:20, 34:43] = 1.0

        symbols = Ocr._split_symbols(matrix)
        self.assertEqual(len(symbols), 2)
        self.assertEqual(symbols[0].shape, (48, 48))
        self.assertEqual(symbols[1].shape, (48, 48))

    def test_recognize_text_uses_model_predictions(self):
        ocr = Ocr()
        ocr._ensure_model = lambda: None
        idx_1 = ocr.char_to_idx["1"]
        idx_zh = ocr.char_to_idx["\u0436"]
        ocr.model = DummyModel([idx_1, idx_zh], len(ocr.CHARSET))

        matrix = np.zeros((48, 60), dtype=np.float32)
        matrix[7:21, 4:12] = 1.0
        matrix[7:21, 15:24] = 1.0
        ocr._prepare_image = lambda _: matrix

        text = ocr.recognize_text("ignored.png")
        self.assertEqual(text, "1\u0436")

    def test_result_returns_empty_for_no_symbols(self):
        ocr = Ocr()
        ocr._ensure_model = lambda: None
        ocr.model = DummyModel([], len(ocr.CHARSET))
        ocr._prepare_image = lambda _: np.zeros((48, 48), dtype=np.float32)

        self.assertEqual(ocr.result("ignored.png"), [])


if __name__ == "__main__":
    unittest.main()
