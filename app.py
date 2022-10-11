# paddleocr loaded from local directory
import sys
sys.path.append('/home/PaddleOCR')

import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from paddleocr import PaddleOCR
import cv2 as cv
import numpy as np

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = "application/json;charset=utf-8"

paddle_ocr = PaddleOCR(use_gpu=False, lang='ch')

def image_template_search(image, template):
    image = np.frombuffer(image, np.uint8)
    template = np.frombuffer(template, np.uint8)

    img_rgb = cv.imdecode(image, 1)
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    template = cv.imdecode(template, 0)
    w, h = template.shape[::-1]
    res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    x, y = list(zip(*loc[::-1]))[0]
    return { 'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h) }

@app.route("/ocr", methods=['GET', 'POST'])
def ocr_route():
    if request.method == 'POST':
        buf = request.files['file'].read()
        result = paddle_ocr.ocr(buf, cls=True)

        return jsonify({'result': result})
    else:
        return render_template('upload.html')

@app.route("/image_template", methods=['GET', 'POST'])
def image_template_route():
    if request.method == 'POST':
        image = request.files['file'].read()
        template = request.files['template'].read()
        result = image_template_search(image, template)

        return jsonify({'result': result})
    else:
        return render_template('image_template.html')

if __name__ == "__main__":
    if os.getenv('FLASK_ENV') == 'production':
        app.run(host='0.0.0.0', port=5000)
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)
