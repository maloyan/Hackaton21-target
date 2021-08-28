 FROM python:3.8
 COPY requirements.txt .
 
 RUN pip install --user -r requirements.txt
 
 WORKDIR /code
 
 COPY . .
 
 RUN python setup.py install 
 
 CMD ["/bin/sh", "-c", "echo \"Приложение запущено: http://0.0.0.0:8989 \"    && python aim_target/webapp.py checkpoints/128_resnet50.onnx"]