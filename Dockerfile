FROM nvcr.io/nvidia/pytorch:21.06-py3
COPY requirements.txt .

RUN pip install -r requirements.txt