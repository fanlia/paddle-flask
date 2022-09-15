FROM registry.baidubce.com/paddlepaddle/paddle:2.1.3-gpu-cuda10.2-cudnn7

WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple

EXPOSE 5000
COPY . .
CMD ["python", "app.py"]
