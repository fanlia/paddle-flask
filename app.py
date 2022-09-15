import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from paddleocr import PaddleOCR

app = Flask(__name__)
CORS(app)

paddle_ocr = PaddleOCR(use_gpu=False, lang='ch')

@app.route("/ocr", methods=['POST'])
def ocr_route():
    file = request.files['file']
    buf = file.read()
    result = paddle_ocr.ocr(buf, cls=True)

    return jsonify({'result': result})

if __name__ == "__main__":
    if os.getenv('FLASK_ENV') == 'production':
        app.run(host='0.0.0.0', port=5000)
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)
