from pathlib import Path

from flask import Flask, render_template, request

from ai.EasyOCR import Ocr

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB

BASE_DIR = Path(__file__).resolve().parent
BASE_IMAGE_DIR = BASE_DIR / "base"
BASE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OCR_AGENT = Ocr()

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _render_page(
    status_text: str = "Загрузите изображение с текстом.",
    result_text: str = "",
    show_result: bool = False,
    file_name: str = "",
):
    return render_template(
        "index.html",
        status_text=status_text,
        result_text=result_text,
        show_result=show_result,
        file_name=file_name,
    )


@app.route("/")
def index():
    return _render_page()


@app.route("/action", methods=["POST"])
def action():
    uploaded_file = request.files.get("file")
    if not uploaded_file or not uploaded_file.filename:
        return _render_page(
            status_text="Сначала выберите изображение",
            show_result=True,
        )

    safe_name = Path(uploaded_file.filename).name
    extension = Path(safe_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return _render_page(
            status_text="Не подходящий формат",
            result_text="Только: png, jpg, jpeg, bmp, webp",
            show_result=True,
            file_name=safe_name,
        )

    image_path = BASE_IMAGE_DIR / safe_name
    uploaded_file.save(image_path)

    try:
        preds = OCR_AGENT.result(safe_name)
    except Exception:
        return _render_page(
            status_text="Ошибка EasyOCR",
            result_text="Проверьте наличие плагина EasyOCR",
            show_result=True,
            file_name=safe_name,
        )

    if not preds:
        return _render_page(
            status_text="Не обнаруженно",
            show_result=True,
            file_name=safe_name,
        )

    return _render_page(
        status_text="Распознаный текст",
        result_text=preds[0],
        show_result=True,
        file_name=safe_name,
    )


if __name__ == "__main__":
    app.run(port=8080, host="127.0.0.1")
