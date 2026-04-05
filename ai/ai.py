import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


class Ocr:
    """TensorFlow/Keras OCR agent for numbers, Russian words, and English words."""

    IMAGE_SIZE = 48
    DIGITS = "0123456789"
    ENG_LOWER = "abcdefghijklmnopqrstuvwxyz"
    ENG_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    RUS_LOWER = (
        "\u0430\u0431\u0432\u0433\u0434\u0435\u0451\u0436\u0437\u0438\u0439"
        "\u043a\u043b\u043c\u043d\u043e\u043f\u0440\u0441\u0442\u0443\u0444"
        "\u0445\u0446\u0447\u0448\u0449\u044a\u044b\u044c\u044d\u044e\u044f"
    )
    RUS_UPPER = (
        "\u0410\u0411\u0412\u0413\u0414\u0415\u0401\u0416\u0417\u0418\u0419"
        "\u041a\u041b\u041c\u041d\u041e\u041f\u0420\u0421\u0422\u0423\u0424"
        "\u0425\u0426\u0427\u0428\u0429\u042a\u042b\u042c\u042d\u042e\u042f"
    )
    CHARSET = DIGITS + ENG_LOWER + ENG_UPPER + RUS_LOWER + RUS_UPPER

    def __init__(self, model_path: str | None = None):
        default_path = Path(__file__).resolve().parent / "text_ocr_model.keras"
        self.model_path = Path(model_path) if model_path else default_path
        self.model = None
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.CHARSET)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.CHARSET)}
        self._font_cache: dict[tuple[str, int], ImageFont.ImageFont] = {}

    @staticmethod
    def _font_candidates() -> list[Path]:
        windows_fonts = Path("C:/Windows/Fonts")
        return [
            windows_fonts / "arial.ttf",
            windows_fonts / "arialbd.ttf",
            windows_fonts / "times.ttf",
            windows_fonts / "timesbd.ttf",
            windows_fonts / "calibri.ttf",
            windows_fonts / "segoeui.ttf",
            windows_fonts / "tahoma.ttf",
            windows_fonts / "verdana.ttf",
            windows_fonts / "consola.ttf",
        ]

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        for font_path in self._font_candidates():
            key = (str(font_path), size)
            if key in self._font_cache:
                return self._font_cache[key]
            if not font_path.exists():
                continue
            try:
                font = ImageFont.truetype(str(font_path), size=size)
                self._font_cache[key] = font
                return font
            except OSError:
                continue
        return ImageFont.load_default()

    def _render_char(self, char: str) -> np.ndarray:
        canvas = Image.new("L", (self.IMAGE_SIZE, self.IMAGE_SIZE), color=255)
        draw = ImageDraw.Draw(canvas)
        font = self._load_font(size=random.randint(24, 40))

        bbox = draw.textbbox((0, 0), char, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        x = max(0, (self.IMAGE_SIZE - width) // 2 + random.randint(-4, 4))
        y = max(0, (self.IMAGE_SIZE - height) // 2 + random.randint(-4, 4))
        draw.text(
            (x, y),
            char,
            font=font,
            fill=random.randint(0, 40),
            stroke_width=random.choice((0, 0, 1)),
            stroke_fill=random.randint(0, 70),
        )

        if random.random() < 0.35:
            canvas = canvas.rotate(
                random.uniform(-10, 10),
                expand=False,
                fillcolor=255,
                resample=Image.Resampling.BILINEAR,
            )
        if random.random() < 0.35:
            canvas = ImageEnhance.Contrast(canvas).enhance(random.uniform(0.8, 1.3))
        if random.random() < 0.2:
            canvas = canvas.filter(ImageFilter.GaussianBlur(random.uniform(0.0, 1.0)))

        arr = np.array(canvas, dtype=np.float32)
        if random.random() < 0.35:
            noise = np.random.normal(0, random.uniform(3.0, 12.0), arr.shape)
            arr = arr + noise
        arr = np.clip(arr, 0, 255) / 255.0
        arr = 1.0 - arr
        return arr

    def _build_synthetic_dataset(self, samples_per_char: int) -> tuple[np.ndarray, np.ndarray]:
        images = []
        labels = []
        for ch, idx in self.char_to_idx.items():
            for _ in range(samples_per_char):
                images.append(self._render_char(ch))
                labels.append(idx)

        x = np.array(images, dtype=np.float32)[..., np.newaxis]
        y = to_categorical(np.array(labels, dtype=np.int32), len(self.CHARSET))
        return x, y

    def create_model(
        self,
        epochs: int = 8,
        batch_size: int = 64,
        samples_per_char: int = 120,
    ) -> bool:
        """Train OCR model for digits + English + Russian characters."""
        try:
            tf.random.set_seed(42)
            random.seed(42)
            np.random.seed(42)

            x, y = self._build_synthetic_dataset(samples_per_char=samples_per_char)
            indices = np.random.permutation(len(x))
            x = x[indices]
            y = y[indices]

            split_idx = int(len(x) * 0.9)
            x_train, x_val = x[:split_idx], x[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            model = models.Sequential(
                [
                    layers.Input(shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 1)),
                    layers.Conv2D(32, (3, 3), activation="relu"),
                    layers.BatchNormalization(),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(64, (3, 3), activation="relu"),
                    layers.BatchNormalization(),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(128, (3, 3), activation="relu"),
                    layers.BatchNormalization(),
                    layers.MaxPooling2D((2, 2)),
                    layers.Dropout(0.3),
                    layers.Flatten(),
                    layers.Dense(256, activation="relu"),
                    layers.Dropout(0.3),
                    layers.Dense(len(self.CHARSET), activation="softmax"),
                ]
            )
            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
            )

            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(self.model_path)
            self.model = model
            return True
        except Exception:
            return False

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. Run create_model() first."
            )
        loaded_model = load_model(self.model_path)
        output_units = int(loaded_model.output_shape[-1])
        if output_units != len(self.CHARSET):
            raise ValueError(
                "Saved model charset is outdated. "
                "Run create_model() to retrain with current RU/EN/0-9 charset."
            )
        self.model = loaded_model

    @staticmethod
    def _prepare_image(image_path: str | Path) -> np.ndarray:
        image = Image.open(image_path).convert("L")
        image = ImageOps.autocontrast(image)
        image = image.filter(ImageFilter.MedianFilter(size=3))

        matrix = np.array(image, dtype=np.float32) / 255.0
        if matrix.mean() > 0.5:
            matrix = 1.0 - matrix

        matrix = np.clip(matrix, 0.0, 1.0)
        matrix[matrix < 0.12] = 0.0
        return matrix

    @staticmethod
    def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        start = None
        for idx, value in enumerate(mask):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                runs.append((start, idx))
                start = None
        if start is not None:
            runs.append((start, len(mask)))
        return runs

    @classmethod
    def _split_lines(cls, matrix: np.ndarray) -> list[np.ndarray]:
        row_ink = matrix.sum(axis=1)
        row_mask = row_ink > max(1.0, matrix.shape[1] * 0.01)
        line_runs = cls._find_runs(row_mask)
        lines = []
        for top, bottom in line_runs:
            if bottom - top < 3:
                continue
            lines.append(matrix[top:bottom, :])
        return lines

    @classmethod
    def _split_line_tokens(cls, line: np.ndarray) -> list[np.ndarray | str]:
        col_ink = line.sum(axis=0)
        col_mask = col_ink > max(0.6, line.shape[0] * 0.04)
        char_runs = [(s, e) for s, e in cls._find_runs(col_mask) if (e - s) >= 2]
        if not char_runs:
            return []

        widths = [end - start for start, end in char_runs]
        median_width = float(np.median(widths)) if widths else 6.0
        space_gap = max(5, int(median_width * 0.65))

        tokens: list[np.ndarray | str] = []
        prev_end = None
        for start, end in char_runs:
            if prev_end is not None:
                gap = start - prev_end
                if gap >= space_gap:
                    tokens.append(" ")

            char_img = line[:, start:end]
            row_ink = char_img.sum(axis=1)
            row_mask = row_ink > max(0.3, char_img.shape[1] * 0.04)
            row_runs = cls._find_runs(row_mask)
            if not row_runs:
                prev_end = end
                continue

            top, bottom = row_runs[0][0], row_runs[-1][1]
            char_img = char_img[top:bottom, :]
            if char_img.shape[0] < 2 or char_img.shape[1] < 2:
                prev_end = end
                continue

            tokens.append(cls._resize_symbol(char_img))
            prev_end = end

        return tokens

    @classmethod
    def _resize_symbol(cls, symbol: np.ndarray) -> np.ndarray:
        symbol = np.clip(symbol, 0.0, 1.0)
        image = Image.fromarray((symbol * 255).astype(np.uint8), mode="L")
        image.thumbnail((36, 36), Image.Resampling.LANCZOS)

        canvas = Image.new("L", (cls.IMAGE_SIZE, cls.IMAGE_SIZE), color=0)
        x_offset = (cls.IMAGE_SIZE - image.width) // 2
        y_offset = (cls.IMAGE_SIZE - image.height) // 2
        canvas.paste(image, (x_offset, y_offset))
        return np.array(canvas, dtype=np.float32) / 255.0

    @classmethod
    def _split_symbols(cls, matrix: np.ndarray) -> list[np.ndarray]:
        """Compatibility helper: returns symbols from the first detected line."""
        lines = cls._split_lines(matrix)
        if not lines:
            return []
        tokens = cls._split_line_tokens(lines[0])
        return [token for token in tokens if isinstance(token, np.ndarray)]

    def _decode_tokens(self, tokens: list[np.ndarray | str]) -> str:
        chars = [token for token in tokens if isinstance(token, np.ndarray)]
        if not chars:
            return ""

        batch = np.array(chars, dtype=np.float32)[..., np.newaxis]
        probs = self.model.predict(batch, verbose=0)
        preds = np.argmax(probs, axis=1)

        out = []
        char_index = 0
        for token in tokens:
            if isinstance(token, str):
                out.append(token)
            else:
                out.append(self.idx_to_char.get(int(preds[char_index]), ""))
                char_index += 1
        return "".join(out).strip()

    def recognize_text(self, image_path: str | Path) -> str:
        self._ensure_model()
        matrix = self._prepare_image(image_path)
        lines = self._split_lines(matrix) or [matrix]

        decoded_lines = []
        for line in lines:
            tokens = self._split_line_tokens(line)
            line_text = self._decode_tokens(tokens)
            if line_text:
                decoded_lines.append(line_text)
        return "\n".join(decoded_lines)

    # Backward-compatible wrappers
    def pred(self, image_path: str | Path) -> np.ndarray:
        text = self.recognize_text(image_path)
        if text.isdigit():
            return np.array([int(ch) for ch in text], dtype=np.int64)
        return np.array(list(text), dtype="<U1")

    def result(self, image_path: str | Path) -> list[str]:
        text = self.recognize_text(image_path)
        return [text] if text else []


if __name__ == "__main__":
    ocr = Ocr()
    if not ocr.model_path.exists():
        print("Training text OCR model (digits + en + ru)...")
        print(ocr.create_model())
    else:
        print(f"Model found: {ocr.model_path}")
