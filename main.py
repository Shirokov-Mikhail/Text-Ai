from flask import Flask, render_template, request
from ai.EasyOCR import Ocr

import numpy as np
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', result='none', result_text='')

@app.route('/action', methods=["POST"])
def acrion():
    img = request.form.get('file')
    print(img)
    ocr = Ocr()
    preds = ocr.result(img)
    return render_template('index.html', result='block', result_text='\n'.join(preds))

if __name__ == '__main__':
    app.run(port=8080, host='127.0.0.1')
