
from pprint import pprint
import random
from flask import Flask, render_template, url_for, request, jsonify, redirect

import json

app = Flask(__name__)

# Required configuratio
auth = False
email = '0'
permissions = 0


@app.route('/')
def index():
    fone = [url_for('static', filename='uploads/posters/poster.jpg') for i in range(3)]
    if auth:
        return render_template('index.html', display="inline", user="none", fone=fone)
    return render_template('index.html', display="none", user="inline", fone=fone)



if __name__ == '__main__':
    app.run(port=8080, host='127.0.0.1')
