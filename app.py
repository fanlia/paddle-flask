import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from paddleocr import PaddleOCR

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = "application/json;charset=utf-8"

paddle_ocr = PaddleOCR(use_gpu=False, lang='ch')

@app.route("/ocr", methods=['GET', 'POST'])
def ocr_route():
    if request.method == 'POST':
        file = request.files['file']
        buf = file.read()
        result = paddle_ocr.ocr(buf, cls=True)

        return jsonify({'result': result})
    else:
        return render_template('upload.html')

if __name__ == "__main__":
    if os.getenv('FLASK_ENV') == 'production':
        app.run(host='0.0.0.0', port=5000)
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)
